import os
import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import markdown
import tqdm


def load_model():
    """Load the Qwen2.5-VL model with optimized settings."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    return model, processor


def generate_blade_report(
        rgb_image_paths, thermal_image_paths, model, processor, turbine_blade_prefix
):
    """
    Generate a comprehensive blade condition report using both RGB and thermal images.

    :param rgb_image_paths: A list of file paths to RGB images for this blade.
    :param thermal_image_paths: A list of file paths to thermal images for this blade.
    :param model: The loaded Qwen2.5-VL model.
    :param processor: The corresponding Qwen2.5-VL processor.
    :param turbine_blade_prefix: The prefix representing this particular turbine + blade.
    """

    # Prepare the system prompt for Qwen
    system_prompt = (
        "You are a senior expert specializing in wind turbine rotor blade inspection.\n"
        "You need to inspect RGB (visual) images and thermal images for the turbine blade.\n"
        "Please generate a comprehensive inspection report that includes (but not limited to):\n"
        " - Observed anomalies or damage (e.g., cracks, erosion, delamination)\n"
        " - Thermal anomalies or hotspots\n"
        " - Severity assessment and potential causes\n"
        " - Recommended maintenance or repair actions\n"
        " - Implications for turbine performance\n"
    )

    # Build the user content by attaching all relevant images
    user_content = []

    # 1) Attach all RGB images
    if rgb_image_paths:
        user_content.append(
            {
                "type": "text",
                "text": f"Below are {len(rgb_image_paths)} RGB images for {turbine_blade_prefix}:"
            }
        )
        for rgb_path in rgb_image_paths:
            user_content.append({"type": "image", "image": rgb_path})

    # 2) Attach all thermal images
    if thermal_image_paths:
        user_content.append(
            {
                "type": "text",
                "text": f"Below are {len(thermal_image_paths)} thermal images for {turbine_blade_prefix}:"
            }
        )
        for thr_path in thermal_image_paths:
            user_content.append({"type": "image", "image": thr_path})

    # Compose messages for Qwen
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Convert the conversation into the format Qwen requires
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # Create inputs for Qwen
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate the report
    generated_ids = model.generate(**inputs, max_new_tokens=1024)

    # Trim off the user prompt IDs
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def group_files_by_turbine_blade(rgb_dir, thermal_dir):
    """
    Scan two directories (RGB images, thermal images) and group files by turbine-blade prefix.

    Example file names:
        - Turbine-2_A_PS_sequence_13.jpg
        - Turbine-6_A_PS_sequence_79.jpg

    We'll assume the "turbine-blade" prefix is something like "Turbine-2_A"
    based on the pattern: r"^(Turbine-\d+_[A-Z])"
    """
    pattern = re.compile(r"^(Turbine-\d+_[A-Z])")

    # Data structure: { "Turbine-2_A": {"rgb": [...], "thermal": [...]} }
    file_groups = {}

    # Process RGB
    if os.path.isdir(rgb_dir):
        for fname in sorted(os.listdir(rgb_dir)):
            # Filter for image files
            if fname.lower().endswith(".jpg") or fname.lower().endswith(".png"):
                match = pattern.match(fname)
                if match:
                    prefix = match.group(1)
                    file_groups.setdefault(prefix, {"rgb": [], "thermal": []})
                    file_groups[prefix]["rgb"].append(os.path.join(rgb_dir, fname))

    # Process thermal
    if os.path.isdir(thermal_dir):
        for fname in sorted(os.listdir(thermal_dir)):
            # Filter for image files in thermal directory
            if fname.lower().endswith(".jpg") or fname.lower().endswith(".png"):
                match = pattern.match(fname)
                if match:
                    prefix = match.group(1)
                    file_groups.setdefault(prefix, {"rgb": [], "thermal": []})
                    file_groups[prefix]["thermal"].append(os.path.join(thermal_dir, fname))

    return file_groups


def process_turbine_data_to_html(rgb_dir, thermal_dir, output_file, model, processor):
    """
    Group all related RGB and thermal image files by turbine-blade prefix,
    generate a report for each group (with markdown formatting), convert them to HTML,
    and save the combined results in an HTML file.
    """
    groups = group_files_by_turbine_blade(rgb_dir, thermal_dir)
    html_sections = []

    for prefix, paths_dict in tqdm.tqdm(groups.items()):
        rgb_files = paths_dict["rgb"]
        thermal_files = paths_dict["thermal"]

        # Skip if no data found
        if not rgb_files and not thermal_files:
            continue

        # Generate the inspection report in markdown format
        report = generate_blade_report(rgb_files, thermal_files, model, processor, prefix)

        # Convert the markdown report to HTML (this converts **text** to <strong>text</strong>, etc.)
        report_html = markdown.markdown(report)

        # Create an HTML section for the current report
        section_html = f"<h2>Report for {prefix}</h2>\n<div>{report_html}</div>"
        print(section_html)  # Optional: print for debugging
        html_sections.append(section_html)

    # Wrap all sections in basic HTML boilerplate
    html_content = (
        "<html>\n"
        "  <head>\n"
        "    <meta charset='UTF-8'>\n"
        "    <title>Turbine Inspection Reports</title>\n"
        "  </head>\n"
        "  <body>\n"
        + "\n".join(html_sections) +
        "\n  </body>\n"
        "</html>"
    )

    # Save the HTML content to the specified file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    # Directories containing your KI-VISIR dataset
    RGB_IMAGE_DIR = "/home/u2370656/PycharmProjects/OffshoreLLM/ki-visir_dataset_v1/combined_images"
    THERMAL_DIR = "/home/u2370656/PycharmProjects/OffshoreLLM/ki-visir_dataset_v1/combined_thermal_images"

    # Output file for the comprehensive blade inspection reports
    OUTPUT_FILE = "blade_inspection_reports.html"

    # Load the model & processor
    model, processor = load_model()

    # Generate reports for all turbines/blades found in the dataset
    process_turbine_data_to_html(RGB_IMAGE_DIR, THERMAL_DIR, OUTPUT_FILE, model, processor)
