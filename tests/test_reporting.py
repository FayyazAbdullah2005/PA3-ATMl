import csv
from pathlib import Path
import tempfile
import unittest
import zipfile
from pa3.io import write_json,read_json
from pa3.reporting import csv_export,summarize


class ReportingTests(unittest.TestCase):
    def test_csv_keeps_unicode_and_escapes_spreadsheet_formulas(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"safe.csv"
            csv_export(path,[{"answer":"α = ½","text":"=HYPERLINK(\"bad\")"}])
            with path.open(encoding="utf-8-sig",newline="") as f:
                row=next(csv.DictReader(f))
            self.assertEqual(row["answer"],"α = ½")
            self.assertTrue(row["text"].startswith("'="))

    def test_exports_are_complete_zip_readable_and_preserve_manual_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"runs"/"synthetic"/"task1-qualitative"
            def response(condition):
                return {"prompt_id":"what_is_your_wish","condition":condition,
                    "generated_text":"<script>alert('unsafe')</script>\n"+"Long response α. "*200,
                    "hit_max_new_tokens":False,"messages":[{"role":"user","content":"Synthetic validation only"}]}
            write_json(source/"baseline"/"responses.json",[response("baseline")])
            summarize(root,"synthetic")
            output=root/"exports"/"synthetic"
            text=(output/"task1_responses.html").read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;",text)
            self.assertNotIn("<script>",text)
            annotation=output/"task1_annotations.csv"
            with annotation.open(encoding="utf-8-sig",newline="") as f:
                rows=list(csv.DictReader(f))
            rows[0]["notes"]="Student annotation retained"
            csv_export(annotation,rows)
            write_json(source/"full"/"responses.json",[response("full")])
            summarize(root,"synthetic")
            with annotation.open(encoding="utf-8-sig",newline="") as f:
                revised=list(csv.DictReader(f))
            self.assertEqual(len(revised),2)
            self.assertEqual(next(r for r in revised if r["condition"]=="baseline")["notes"],"Student annotation retained")
            self.assertFalse(read_json(output/"inventory.json")["complete_counts"])
            with zipfile.ZipFile(root/"exports"/"synthetic.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("index.html",archive.namelist())
                self.assertIn("task1_annotations.csv",archive.namelist())


if __name__=="__main__":
    unittest.main()

