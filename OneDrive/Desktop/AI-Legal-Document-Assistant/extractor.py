import PyPDF2

def extract_text(file):

    text = ""

    try:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    except Exception as e:
        print("Error reading PDF:", e)

    return text