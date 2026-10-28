import os
import win32com.client

docx_path = os.path.abspath("SI_Lab_Project_Report_Population_Data_Analysis.docx")
pdf_path = os.path.abspath("SI_Lab_Project_Report_Population_Data_Analysis.pdf")

print(f"Converting {docx_path} to {pdf_path}...")
word = win32com.client.Dispatch("Word.Application")
word.Visible = False

try:
    doc = word.Documents.Open(docx_path)
    # wdFormatPDF = 17
    doc.SaveAs(pdf_path, FileFormat=17)
    doc.Close()
    print("SUCCESS: PDF created successfully.")
except Exception as e:
    print("Error during PDF conversion:", e)
finally:
    word.Quit()

