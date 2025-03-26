# 🌀 LLM-Blade: Automatic Wind Turbine Blade Inspection Report Generation

This repository contains the code and results from our study on using Large Language Models (LLMs) to automatically generate wind turbine blade inspection reports directly from RGB (visual) and thermal images.

We explore a simple, end-to-end approach where images and structured prompts are passed to multimodal LLMs to generate human-readable inspection summaries.

## 🔍 What This Repository Includes

- ✅ Code for generating structured prompts and interfacing with multimodal LLMs.
- ✅ Preprocessed RGB and thermal image inputs (where permitted).
- ✅ Automatically generated reports for 90 turbine blades using the [Qwen LLM](https://huggingface.co/Qwen).
- ✅ Example system and user prompts used for report generation.
- ✅ A sample evaluation script for organizing reports by turbine and image metadata.
- ✅ Guidelines for reviewing and correcting reports for future fine-tuning.

## 📁 Repository Structure

```
LLM-Blade/
├── reports/               # Generated inspection reports (Qwen, 90 blades)
├── prompts/               # Prompt templates used in the study
├── images/                # Input RGB and thermal images (subset, if allowed)
├── src/                   # Source code for generating and organizing reports
│   ├── run_qwen.py        # Main script to run report generation with Qwen
│   └── utils.py           # Utility functions
├── examples/              # Selected report examples for demonstration
├── evaluation/            # Tools for organizing and comparing reports
└── README.md              # This file
```

## 📌 Usage

> Requirements:
- Python 3.8+
- Access to the Qwen LLM (via Hugging Face or other interface)
- `transformers`, `torch`, `Pillow`, and other standard packages

To generate reports from a folder of images:

```bash
python src/run_qwen.py --input_dir ./images/ --output_dir ./reports/
```

Customize the prompts in `prompts/system_prompt.txt` and `prompts/user_prompt_template.txt` as needed.

## 📤 Contributing Expert Feedback

We welcome domain experts to review and correct generated reports. Please see the `reports/` folder and use the provided template in `evaluation/feedback_template.md` to suggest improvements.

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

## 📫 Contact

For questions or contributions, feel free to open an issue or reach out via GitHub Discussions.

