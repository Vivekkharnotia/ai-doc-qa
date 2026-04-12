import fitz
import tempfile
import os
from fastapi import UploadFile


class PDFService:

    async def extract_text(self, file: UploadFile):
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        # Open PDF using file path
        doc = fitz.open(temp_path)

        text = ""
        for page in doc:
            text += page.get_text()

        doc.close()

        # Delete temp file
        os.remove(temp_path)

        return text
