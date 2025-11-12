import os
import re
import json
import pandas as pd
from functools import partial
from PIL import Image

from ..image_base import ImageBaseDataset, img_root_map
from ..utils import build_judge, DEBUG_MESSAGE
from ...smp import *
from ...utils import track_progress_rich
from ipdb import set_trace as st

logger = get_logger("RUN")

# Official AITZ prompt template from template.py
AITZ_PROMPT = """You are GUI-Long, a reasoning GUI Agent Assistant. Given a task instruction, a screen observation, and an action history sequence, I want you to continue executing the task instruction.
Here is the action space:
1. `CLICK`: Click on an element, value is not applicable and the point [x,y] is required.
2. `TYPE`: Type a string into an element, value is a string to type and the point is not applicable.
3. `SCROLL`: Scroll the screen, choose 'UP' 'DOWN' 'LEFT' 'RIGHT' as your value to indicate the direction you want the view to move.
4. `PRESS`: Execute a press operation. The value can be 'BACK', 'HOME', or 'ENTER', corresponding to returning to the previous step, navigating to the home screen, or submitting the input, point is not applicable.
5. `COMPLETE`: Indicate the task is completed, value and point are not applicable.
I want you to continue executing the instruction `{instruction}`, with the action history being `{history}`, And the current UI screenshot is <image>, .
Please provide the action to perform (enumerate from ['CLICK', 'TYPE', 'SCROLL', 'PRESS', 'COMPLETE']), the point where the cursor is moved to (integer) if a click is performed, and any text required to complete the action.

Output the thinking process in </think> ... </think> tags, a short summary of what it's about to do at this step in <summarized> ... </summarized>, and the final answer in <answer> </answer> tags as follows:
</think> ... </think> <summarized> ... </summarized> <answer>[{{'action': enum['CLICK', 'TYPE', 'SCROLL', 'PRESS', 'COMPLETE'], 'point': [x, y], 'value': 'None[default]'}}]</answer>
If value is not applicable, set it as 'None', If point is not applicable, set it as [-100, -100].
point represents the relative coordinates on the screenshot and should be scaled to a range of 0-1000.
Note:
specific value (no default) is necessary for actions enum['SCROLL', 'TYPE', 'PRESS']
Example:
[{{'action': enum['CLICK'], 'point': [123, 300], 'value': 'None'}}]
[{{'action': enum['TYPE'], 'point': [-100, -100], 'value': 'shanghai shopping mall'}}]
[{{'action': enum['SCROLL'], 'point': [-100, -100], 'value': enum['UP', 'LEFT', 'RIGHT', 'DOWN']}}]
[{{'action': enum['PRESS'], 'point': [-100, -100], 'value': enum['BACK', 'HOME', 'ENTER']}}]
[{{'action': enum['COMPLETE'], 'point': [-100, -100], 'value': 'None'}}]"""


def parse_action_response(response):
    """
    Parse the model response to extract action, point, and value.
    Handles AITZ format with <answer> tags.

    Args:
        response (str): Model response

    Returns:
        dict: Parsed action with keys 'action', 'point', 'value'
    """
    # First, try to extract content from <answer> tags (AITZ format)
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)

    if answer_match:
        answer_content = answer_match.group(1).strip()
        # Try to extract JSON array from answer content
        json_match = re.search(r'\[.*?\]', answer_content, re.DOTALL)
        if json_match:
            action_str = json_match.group(0)
        else:
            action_str = answer_content
    else:
        # If no <answer> tags, try to extract JSON array directly from response
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            action_str = json_match.group(0)
        else:
            action_str = response

    # Clean up the action string
    action_str = action_str.strip()

    # Try to parse as JSON
    try:
        # Replace single quotes with double quotes for JSON compatibility
        action_str = action_str.replace("'", '"')

        # Remove any trailing commas before closing brackets/braces
        action_str = re.sub(r',(\s*[}\]])', r'\1', action_str)

        # Fix missing commas between elements
        action_str = re.sub(r'"\s+"', '", "', action_str)  # Between strings
        action_str = re.sub(r'\]\s*\[', '],[', action_str)  # Between arrays
        action_str = re.sub(r'}\s*{', '},{', action_str)  # Between objects
        action_str = re.sub(r'(\d+)\s+(\d+)', r'\1, \2', action_str)  # Between numbers in arrays

        action_list = json.loads(action_str)
        if action_list and len(action_list) > 0:
            result = action_list[0]
            if isinstance(result, dict):
                return result
            else:
                logger.warning(f"Action is not a dictionary: {type(result)} - {result}")
    except Exception as e:
        logger.warning(f"Failed to parse action response as JSON: {e}")
        logger.debug(f"Action string that failed to parse: {action_str}")

    # If JSON parsing fails, try to extract information using regex
    try:
        # Extract action
        action_match = re.search(r"'action':\s*'([^']*)'", response.replace('"', "'"))
        if not action_match:
            action_match = re.search(r'"action":\s*"([^"]*)"', response)

        # Extract point
        point_match = re.search(r"'point':\s*\[([^\]]*)\]", response.replace('"', "'"))
        if not point_match:
            point_match = re.search(r'"point":\s*\[([^\]]*)\]', response)

        # Extract value
        value_match = re.search(r"'value':\s*'([^']*)'", response.replace('"', "'"))
        if not value_match:
            value_match = re.search(r'"value":\s*"([^"]*)"', response)

        if action_match:
            action = action_match.group(1).upper()

            # Parse point
            if point_match:
                point_str = point_match.group(1)
                try:
                    point = [int(x.strip()) for x in point_str.split(',')]
                except:
                    point = [-100, -100]
            else:
                point = [-100, -100]

            # Parse value
            value = value_match.group(1) if value_match else 'None'

            return {
                'action': action,
                'point': point,
                'value': value
            }
    except Exception as e:
        logger.warning(f"Failed to parse action response using regex: {e}")

    # Fallback: return a default action
    logger.warning(f"Using default action for response: {response[:200]}...")
    return {
        'action': 'UNKNOWN',
        'point': [-100, -100],
        'value': 'None'
    }


def compute_action_accuracy(pred_action, gt_action):
    """
    Compute accuracy between predicted and ground truth actions.

    Args:
        pred_action (dict): Predicted action
        gt_action (dict): Ground truth action

    Returns:
        dict: Accuracy metrics
    """
    # Compare action type
    action_match = pred_action['action'].upper() == gt_action['action'].upper()

    # Compare point (with tolerance)
    pred_point = pred_action.get('point', [-100, -100])
    gt_point = gt_action.get('point', [-100, -100])
    if gt_point[0]<0:
        gt_point==[-100, -100]
    point_match = False
    if pred_point == [-100, -100] and gt_point == [-100, -100]:
        point_match = True  # Both are invalid points
    else:
        # Normalize coordinates to the same range before comparing
        # GT coordinates are normalized (0-1), convert to 0-1000 range to match PRED
        if gt_point[0] <= 1 and gt_point[1] <= 1:
            # GT is normalized, convert to 0-1000 range
            gt_point_normalized = [gt_point[0] * 1000, gt_point[1] * 1000]
        else:
            # GT is already in 0-1000 range
            gt_point_normalized = gt_point

        # Check if points are close (within 50 pixels)
        distance = ((pred_point[0] - gt_point_normalized[0])**2 + (pred_point[1] - gt_point_normalized[1])**2)**0.5
        point_match = distance <= 100

    # Compare text value
    pred_value = str(pred_action.get('value', '')).upper()
    gt_value = str(gt_action.get('value', '')).upper()
    text_match = pred_value == gt_value

    return {
        'action_match': action_match,
        'point_match': point_match,
        'text_match': text_match,
        'overall_match': action_match and point_match and text_match
    }


# def normalize_point(point, image_size):
#     """
#     Normalize point coordinates to [0, 1] range.

#     Args:
#         point (list): [x, y] coordinates
#         image_size (tuple): (width, height) of the image

#     Returns:
#         list: Normalized [x, y] coordinates
#     """
#     if point[0] > 1 or point[1] > 1:
#         return [point[0] / image_size[0], point[1] / image_size[1]]
#     return point


class AITZ(ImageBaseDataset):
    """
    AITZ (Android in The Zoo) Dataset for Android UI automation tasks.
    This dataset contains screenshots of Android apps along with instructions
    and expected actions for UI automation.
    """
    MODALITY = "IMAGE"
    TYPE = "GUI"

    # DATASET_URL = {
    #     "AITZ_VAL": "http://opencompass.openxlab.space/utils/benchmarks/GUI/AITZ/AITZ_VAL.tsv",
    #     "AITZ_TEST": "http://opencompass.openxlab.space/utils/benchmarks/GUI/AITZ/AITZ_TEST.tsv",
    # }

    # DATASET_MD5 = {
    #     'AITZ_VAL': 'md5_hash_here',  # Replace with actual MD5
    #     'AITZ_TEST': 'md5_hash_here',  # Replace with actual MD5
    # }

    def __init__(
        self,
        dataset="AITZ_VAL",
        skip_noimg=True,
        skeleton=False,
        local_data_path="/disk/zdata1/home/zhangqingyu/work_guo/aitz_sft_test_detail_history.json",
    ):
        ROOT = LMUDataRoot()
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, "images", self.dataset_name)
        self.local_data_path = local_data_path

        if skeleton:
            return

        # Load data from local JSON file
        data = self.load_local_json(local_data_path)
        self.skip_noimg = skip_noimg
        # Filter out items without image_path
        if skip_noimg and "image_path" in data:
            data = data[data["image_path"] != ""]
            data = data[~pd.isna(data["image_path"])]

        data["index"] = [str(idx + 1) for idx in range(len(data))]

        # Since we use image_path instead of image, we set meta_only to False
        # This means the images are stored as paths, not as base64 encoded data
        self.meta_only = False
        self.parse_response_func = parse_action_response

        self.data = data

    def load_local_json(self, json_path):
        """
        Load data from local JSON file and convert to DataFrame format expected by VLMEvalKit.

        Args:
            json_path (str): Path to the local JSON file

        Returns:
            pd.DataFrame: Converted data
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Convert JSON data to DataFrame
        df_data = []

        for idx, item in enumerate(json_data):
            row = {
                'index': str(idx + 1),
                'instruction': item.get('instruction', ''),
                'image_path': item.get('images', [''])[-1] if item.get('images') else '',  # Use image_path field for the last image
                'images': item.get('images', []),  # Keep the full images list for history processing
            }

            # Extract action information
            action = item.get('action', 'UNKNOWN')
            point = item.get('point', [-100, -100])
            text = item.get('text', 'None')

            # Add action-related columns
            row['action'] = action
            row['point'] = str(point)  # Convert to string for TSV compatibility
            row['text'] = text

            # Add optional fields
            if 'history' in item:
                row['history'] = item['history']
            if 'lowlevel_instruction' in item:
                row['lowlevel_instruction'] = item['lowlevel_instruction']
            if 'type' in item:
                row['type'] = item['type']

            df_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(df_data)

        return df

    @classmethod
    def supported_datasets(cls):
        # Support both remote datasets and local dataset
        return ['AITZ_VAL', 'AITZ_TEST', 'AITZ_LOCAL']

    @classmethod
    def get_action_space(cls):
        return "['CLICK', 'TYPE', 'SCROLL', 'COMPLETE', 'PRESS']"

    @classmethod
    def get_trajectory(self, line):
        traj_dict = {
            "task": line["instruction"],
            "lowlevel_instruction": line.get("lowlevel_instruction", ""),
        }
        if "history" in line and line["history"]:
            traj_dict["history"] = line["history"]
        return traj_dict

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Determine if we have history
        has_history = "history" in line and line["history"]

        # Get the images list from the original data
        images_list = line.get("images", [])

        # Build messages in VLMEvalKit format
        msgs = []

        if has_history:
            history_text = line["history"]
            history_images = []

            # Extract history images for <image> tags
            # images_list format: [hist1, hist2, ..., hist_n, current]
            # We need the last two history images:倒数第三张 and 倒数第二张
            total_images = len(images_list)

            if total_images >= 3:
                history_images.append(images_list[-3])  # 倒数第三张 -> 第一个<image>
            if total_images >= 2:
                history_images.append(images_list[-2])  # 倒数第二张 -> 第二个<image>

            # Add history images in order (corresponding to <image> tags in history)
            for img_path in history_images:
                absolute_path = self._get_absolute_image_path(img_path)
                msgs.append(dict(type="image", value=absolute_path))
                
            # Add current image first (last image in the sequence)
            current_img_path = line.get("image_path", "")
            if current_img_path:
                absolute_path = self._get_absolute_image_path(current_img_path)
                msgs.append(dict(type="image", value=absolute_path))
            # Use official AITZ template with <image> tags preserved
            user_instruction = AITZ_PROMPT.format(
                instruction=line["instruction"],
                history=history_text  # Keep <image> tags in history
            )

            # Add text message
            msgs.append(dict(type="text", value=user_instruction))
        else:
            # For cases without history
            user_instruction = AITZ_PROMPT.format(
                instruction=line["instruction"],
                history=""
            )

            # Add current image
            current_img_path = line.get("image_path", "")
            if current_img_path:
                absolute_path = self._get_absolute_image_path(current_img_path)
                msgs.append(dict(type="image", value=absolute_path))

            # Add text message
            msgs.append(dict(type="text", value=user_instruction))

        return msgs

    def _get_absolute_image_path(self, img_path):
        """
        Get the absolute path for an image.

        Args:
            img_path (str): Relative image path (e.g., 'aitz/aitz_images/calculator/episode_0/step_0.png')

        Returns:
            str: Absolute path to the image
        """
        if not img_path or img_path == 'None':
            return img_path

        # If it's already an absolute path, return as is
        if osp.isabs(img_path):
            return img_path

        # If it starts with aitz/, remove the prefix and join with base directory
        if img_path.startswith('aitz/'):
            # Remove the 'aitz/' prefix (5 characters)
            relative_path = img_path[5:]
            # Construct absolute path based on the data directory
            base_dir = '/disk/zdata1/home/zhangqingyu/work_guo/data/android_in_the_zoo/'
            absolute_path = osp.join(base_dir, relative_path)
            return absolute_path

        # For other relative paths, assume they're from the data directory
        base_dir = '/disk/zdata1/home/zhangqingyu/work_guo/data/android_in_the_zoo/'
        absolute_path = osp.join(base_dir, img_path)
        return absolute_path

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate AITZ predictions.
        """
        # Initialize statistics
        stats = {
            'total': 0,
            'action_correct': 0,
            'point_correct': 0,
            'text_correct': 0,
            'overall_correct': 0,
            'format_error': 0
        }

        # Store instance-level results
        results = []

        data = load(eval_file)
        required_columns = ["action", "point", "text", "prediction"]
        for col in required_columns:
            assert col in data, f"Missing required column: {col}"

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        for i in tqdm(range(len(lines))):
            line = lines[i]
            stats['total'] += 1

            # Parse ground truth action
            gt_text = line.get('text', 'None')
            # Handle NaN values by converting them to 'None' string
            if pd.isna(gt_text):
                gt_text = 'None'

            gt_action = {
                'action': line.get('action', 'UNKNOWN'),
                'point': eval(line['point']) if isinstance(line['point'], str) else line['point'],
                'value': gt_text  # Add 'value' field for consistency with pred_action
            }

            # Parse predicted action
            prediction = str(line["prediction"])
            try:
                pred_action = self.parse_response_func(prediction)
                # Ensure pred_action is a dictionary
                if not isinstance(pred_action, dict):
                    logger.warning(f"pred_action is not a dictionary: {type(pred_action)} - {pred_action}")
                    pred_action = {
                        'action': 'UNKNOWN',
                        'point': [-100, -100],
                        'value': 'None'
                    }
                    is_format_error = True
                    stats['format_error'] += 1
                else:
                    is_format_error = False
            except Exception as e:
                logger.warning(f"Failed to parse prediction: {e}")
                pred_action = {
                    'action': 'UNKNOWN',
                    'point': [-100, -100],
                    'value': 'None'
                }
                is_format_error = True
                stats['format_error'] += 1

            # Compute accuracy metrics
            accuracy_metrics = compute_action_accuracy(pred_action, gt_action)

            # Update statistics
            if accuracy_metrics['action_match']:
                stats['action_correct'] += 1
            if accuracy_metrics['point_match']:
                stats['point_correct'] += 1
            if accuracy_metrics['text_match']:
                stats['text_correct'] += 1
            if accuracy_metrics['overall_match']:
                stats['overall_correct'] += 1

            # Store result
            result = {
                "index": i,
                "instruction": line.get("instruction", ""),
                "image": line.get("image", ""),
                "gt_action": gt_action,
                "pred_action": pred_action,
                "accuracy_metrics": accuracy_metrics,
                "is_format_error": is_format_error,
                "prediction": prediction
            }

            # Add additional fields if available
            if "history" in line:
                result["history"] = line["history"]
            if "lowlevel_instruction" in line:
                result["lowlevel_instruction"] = line["lowlevel_instruction"]
            if "type" in line:
                result["type"] = line["type"]

            results.append(result)

        # Calculate final scores
        final_score_dict = {}
        if stats['total'] > 0:
            final_score_dict['Total'] = stats['total']
            final_score_dict['Action_Accuracy'] = (stats['action_correct'] / stats['total']) * 100
            final_score_dict['Point_Accuracy'] = (stats['point_correct'] / stats['total']) * 100
            final_score_dict['Text_Accuracy'] = (stats['text_correct'] / stats['total']) * 100
            final_score_dict['Overall_Accuracy'] = (stats['overall_correct'] / stats['total']) * 100
            final_score_dict['Format_Error_Rate'] = (stats['format_error'] / stats['total']) * 100
        else:
            final_score_dict = {
                'Total': 0,
                'Action_Accuracy': 0,
                'Point_Accuracy': 0,
                'Text_Accuracy': 0,
                'Overall_Accuracy': 0,
                'Format_Error_Rate': 0
            }

        # Save detailed results
        score_pth = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(final_score_dict, score_pth)

        # Save instance-level results
        results_pth = get_intermediate_file_path(eval_file, '_results', 'json')
        dump(results, results_pth)

        # Save failure cases if environment variable is set
        failure_cases_path = os.environ.get("FAILURE_CASES_PATH", None)
        if failure_cases_path is not None:
            failure_cases = [
                res for res in results
                if not res['accuracy_metrics']['overall_match'] or res['is_format_error']
            ]
            with open(failure_cases_path, "w") as f:
                json.dump(failure_cases, f, indent=4, ensure_ascii=False)

        # Save successful cases if environment variable is set
        successful_cases_path = os.environ.get("SUCCESSFUL_CASES_PATH", None)
        if successful_cases_path is not None:
            successful_cases = [
                res for res in results
                if res['accuracy_metrics']['overall_match'] and not res['is_format_error']
            ]
            with open(successful_cases_path, "w") as f:
                json.dump(successful_cases, f, indent=4, ensure_ascii=False)

        return final_score_dict