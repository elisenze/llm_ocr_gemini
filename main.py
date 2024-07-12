import argparse
import os
import logging
import timeit
from typing import Dict

import json
from pdf2image import convert_from_path, pdfinfo_from_path
import google.generativeai as genai

logger = logging.getLogger("Fullerton-OCR")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
logger.addHandler(sh)
logger.info("Logger is set up.")


class DocToExtract:
    """
    A class to represent a document to extract.
    """

    def __init__(
        self,
        pdf_to_extract: str,
        pdf_dir: str,
        output_dir: str,
    ) -> None:

        self.doc_id: str = ""
        self.doc_file_path: str = pdf_to_extract
        self.pdf_dir: str = pdf_dir
        self.output_dir: str = self.create_output_subdir_for_file(output_dir)
        self.num_pages: Dict = pdfinfo_from_path(pdf_to_extract)["Pages"]

    def create_output_subdir_for_file(self, output_dir: str) -> str:
        sub_dir_path = f"{self.doc_file_path.split('/')[-1].split('.')[0]}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if sub_dir_path not in os.listdir(output_dir):
            os.makedirs(os.path.join(output_dir, sub_dir_path))

        output_dir = os.path.join(output_dir, sub_dir_path)
        logger.info(f"output_dir: {output_dir}")
        return output_dir

    def parse_pages_to_extract(self, pages_to_extract: str) -> None:
        """
        Extracts a list of page numbers from a given string containing page numbers or ranges.

        Args:
        pages_to_extract (str): A string containing page numbers or ranges, separated by commas.
        Ex: "1", "1, 2, 3", "1-3",  default="all"

        Returns:
        list: A list of unique page number keys in ascending order. If a
            parsing error occurs, an empty list is returned.

        Return: ['page_1', 'page_2',...]
        """
        if type(pages_to_extract) == list:
            return [f"page_{page}" for page in pages_to_extract]

        pages_to_extract = pages_to_extract.strip(",").strip(" ").strip("-")

        if pages_to_extract.strip().lower() == "all":
            page_list = [page + 1 for page in range(self.num_pages)]

        elif "-" in pages_to_extract:
            pages = pages_to_extract.split("-")
            page_list = [page for page in range(int(pages[0]), int(pages[1]) + 1)]

        elif "," in pages_to_extract:
            pages = pages_to_extract.split(",")
            page_list = [int(page) for page in pages]

        # check if pages_to_extract is a single page number
        else:
            try:
                page_list = [int(pages_to_extract)]
            except ValueError:
                page_list = []
        if page_list:
            page_list = sorted(list(set(page_list)))

        self.pages_to_extract = [f"page_{page}" for page in page_list]

    def get_page_images(self) -> None:
        """
        Extract ONLY images from PDF file needed for extraction.

        Returns:
            list: A list of extracted images as NumPy arrays (OpenCV format).
        """
        pages_to_extract_images_pil = dict()

        if not self.pages_to_extract:
            logger.info("No pages to extract.")

        logger.info(f"Processed pages_to_extract: {self.pages_to_extract}")

        first_page = min([int(page.split("_")[-1]) for page in self.pages_to_extract])
        last_page = max([int(page.split("_")[-1]) for page in self.pages_to_extract])

        pil_images = convert_from_path(
            self.doc_file_path, first_page=first_page, last_page=last_page
        )

        for page_key, pil_image in zip(self.pages_to_extract, pil_images):
            pages_to_extract_images_pil[page_key] = pil_image

        logger.info(
            f"Extracted {len(pages_to_extract_images_pil)} images from {self.doc_file_path}."
        )
        self.pages_to_extract_images_pil = pages_to_extract_images_pil

    def process_with_gemini(self, model) -> None:
        """
        Process the extracted images with Gemini OCR.
        """
        model = setup_gemini()
        prompts = get_prompts()

        doc_response = dict()
        for page_key, pil_image in self.pages_to_extract_images_pil.items():
            page_response = model.generate_content(
                [prompts["prompt"], pil_image]
            ).to_dict()

            page_json = eval(
                page_response["candidates"][0]["content"]["parts"][0]["text"]
            )

            doc_response[page_key] = page_json

        self.response = doc_response


api_key_path = "api/api_key.txt"


def load_api_key(API_Path: str) -> str:
    with open(API_Path, "r") as file:
        api_key = file.read().strip()
        return api_key


def setup_gemini() -> genai.GenerativeModel:
    api_key_source = load_api_key(api_key_path)
    genai.configure(api_key=api_key_source)
    generation_config = get_generation_config()
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    return model


def get_prompts() -> Dict:
    with open("prompts.json", "r") as f:
        prompts = json.load(f)
    return prompts


def get_generation_config() -> Dict:
    with open("gemini_config.json", "r") as f:
        generation_config = json.load(f)
    return generation_config


if __name__ == "__main__":
    start_time = timeit.default_timer()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, default="data/")
    parser.add_argument("--pdf_file_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--pages_to_extract", type=str, default="all")

    args = parser.parse_args()

    model = setup_gemini()

    document = DocToExtract(
        pdf_to_extract=args.pdf_file_path,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
    )

    document.parse_pages_to_extract(args.pages_to_extract)
    document.get_page_images()

    try:
        document.process_with_gemini(model)
    except Exception as e:
        logger.error(f"Error processing document: {e}")

    if hasattr(document, "response"):
        logger.info(f"Gemini response: {document.response}")
        with open(
            os.path.join(document.output_dir, "response.json"), "w"
        ) as response_file:
            json.dump(document.response, response_file)

    elapsed = timeit.default_timer() - start_time

    logger.info(f"Elapsed time: {elapsed:.2f} seconds.")
