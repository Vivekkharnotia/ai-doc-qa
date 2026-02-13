import fitz
from fastapi import UploadFile
import shutil
import os


class PDFService:

    @staticmethod
    def save_uploaded_file(file: UploadFile) -> str:
        """
        Save uploaded file temporarily and return file path.
        """
        file_location = f"temp_{file.filename}"

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_location

    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text from PDF file.
        """
        text = ""

        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()

        return text

    @staticmethod
    def delete_file(file_path: str):
        """
        Remove temporary file.
        """
        if os.path.exists(file_path):
            os.remove(file_path)
