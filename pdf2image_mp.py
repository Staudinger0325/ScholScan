import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path
from tqdm import tqdm  

SRC_DIR = Path("")
OUT_ROOT = Path("")
DPI = 300

def process_pdf(pdf_path: Path) -> tuple[str, int, str | None]:
    try:
        base = pdf_path.stem
        out_dir = OUT_ROOT / base
        out_dir.mkdir(parents=True, exist_ok=True)

        images = convert_from_path(
            pdf_path.as_posix(),
            dpi=DPI,
            fmt="png",
            thread_count=20
        )

        for idx, img in enumerate(images, start=1):
            out_name = f"{base}_{idx}.png"
            out_path = out_dir / out_name
            img.save(out_path.as_posix(), "PNG")

        return (base, len(images), None)
    except Exception as e:
        return (pdf_path.stem, 0, str(e))

def main():
    if not SRC_DIR.is_dir():
        raise FileNotFoundError(f"Origin directory not found: {SRC_DIR}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pdfs = [p for p in SRC_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdfs:
        print("No PDF found.")
        return

    max_workers = min(20, (os.cpu_count() or 4) * 2)

    print(f"Start rendering: {len(pdfs)} PDFs, DPI={DPI}，threads={max_workers}")
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_pdf, pdf): pdf for pdf in pdfs}
        with tqdm(total=len(futures), desc="Converting PDFs", unit="file") as pbar:
            for fut in as_completed(futures):
                base, pages, err = fut.result()
                if err:
                    tqdm.write(f"[Failed] {base}.pdf -> {err}")
                else:
                    tqdm.write(f"[Successed] {base}.pdf -> {pages} 张")
                results.append((base, pages, err))
                pbar.update(1)

    ok = sum(1 for _, _, e in results if e is None)
    fail = len(results) - ok
    total_pages = sum(p for _, p, e in results if e is None)
    print(f"\nFinished: Success {ok}, Fail {fail}, Totally {total_pages}.")
    print(f"output directory: {OUT_ROOT}")

if __name__ == "__main__":
    main()
