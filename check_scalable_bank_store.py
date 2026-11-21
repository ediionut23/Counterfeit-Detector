"""Small integrity checks for normalized identities and resumable imports."""
import csv
import tempfile
from pathlib import Path

from scalable_bank_store import BankStore


def main():
    with tempfile.TemporaryDirectory(prefix="bank_store_check_") as folder:
        root=Path(folder)
        source=root/"input.csv"
        with source.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["molecule_id","smiles","label","generation_method"])
            writer.writeheader()
            writer.writerows([
                {"molecule_id":"a","smiles":"CC(=O)O","label":"0","generation_method":""},
                {"molecule_id":"b","smiles":"CC(=O)[O-].[Na+]","label":"1","generation_method":"M"},
                {"molecule_id":"bad","smiles":"badsmiles","label":"1","generation_method":"E"},
            ])
        bank=BankStore(root/"test.sqlite")
        bank.import_catalogue(source,workers=2,chunk_size=1,holdout={})
        assert bank.count()==1,bank.summary()
        summary=bank.summary()
        assert summary["molecules_with_both_origins"]==1 and summary["method_origin_counts"]=={"M":1}
        before=bank.connection.execute("SELECT COUNT(*) FROM molecule_sources").fetchone()[0]
        bank.import_catalogue(source,workers=2,chunk_size=1,holdout={})
        assert bank.count()==1 and bank.connection.execute("SELECT COUNT(*) FROM molecule_sources").fetchone()[0]==before
        assert bank.connection.execute("SELECT errors FROM import_sources").fetchone()[0]==1
        with source.open("a") as stream:
            stream.write("changed,CC,0,\n")
        try:
            bank.import_catalogue(source,workers=2,chunk_size=1,holdout={})
        except ValueError:
            pass
        else:
            raise AssertionError("Changed source must not silently resume")
        bank.connection.close()
    print("PASS: normalized identity dedup; both-origin provenance; error accounting; resume idempotence; source mutation guard")


if __name__=="__main__":
    main()
