"""Compatibility entry point; use run_text_to_image.py for new experiments."""

from omnimia.pathways.image import main


if __name__ == "__main__":
	main(task_override="t2i")
