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
├── LICENSE                     # MIT License
├── README.md                   # Project overview and instructions
├── blade_inspection_reports.html  # Viewer for generated blade inspection reports
├── feedback_form.html          # Web form to collect expert feedback
├── qwen_report.py              # Script to generate inspection reports using Qwen2.5-VL
```

## 📌 Usage

> Requirements:
- Python 3.8+
- Access to the Qwen LLM (via Hugging Face or other interface)
- `transformers`, `torch`, `Pillow`, and other standard packages

To generate reports, revise the image path in the qwen_report.py and run:

```bash
python qwen_report.py
```

## 📤 Contributing Expert Feedback

We welcome domain experts to review and help improve our generated reports.

📁 Please check the `reports/` folder for the generated content.  
📝 To suggest corrections or enhancements, kindly use our feedback form:  
[**Feedback Form**](https://lironui.github.io/LLM-Blade/feedback_form.html)

Your insights are invaluable in making this project more accurate and impactful!

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

## 📫 Contact

For questions or contributions, feel free to open an issue or reach out via GitHub Discussions.

