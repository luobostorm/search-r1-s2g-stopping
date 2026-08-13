SHELL := /bin/sh

.PHONY: all figures check draft-pdf final-pdf final-pdf-zh check-isolated clean

TECTONIC ?= tectonic
OUTPUT_DIR ?= ../../output/pdf
ZH_OUTPUT_DIR ?= ../../output/pdf-zh
ENGLISH_MAIN ?= main-en.tex
CHINESE_MAIN ?= main.tex

all: final-pdf

figures:
	python3 ../../scripts/generate_s2g_paper_figures.py

check:
	@python3 -c 'import pathlib,re; from collections import Counter; bib=pathlib.Path("references.bib").read_text(encoding="utf-8"); keys=set(re.findall(r"@[A-Za-z]+\{([^,]+),",bib)); files=(pathlib.Path("$(ENGLISH_MAIN)"),pathlib.Path("$(CHINESE_MAIN)")); [(lambda t,p: (not (t.count("{")==t.count("}")) and (_ for _ in ()).throw(AssertionError(str(p)+": 花括号数量不匹配")), not (Counter(re.findall(r"\\begin\{([^}]+)\}",t))==Counter(re.findall(r"\\end\{([^}]+)\}",t))) and (_ for _ in ()).throw(AssertionError(str(p)+": 环境不配对")), not set(sum((re.split(r"\s*,\s*",x) for x in re.findall(r"\\cite[pt]?\{([^}]+)\}",t)),[])).issubset(keys) and (_ for _ in ()).throw(AssertionError(str(p)+": 存在未定义 BibTeX 引用")), ("```" in t or re.search(r"^#{1,6} ",t,re.M)) and (_ for _ in ()).throw(AssertionError(str(p)+": 残留 Markdown 语法")), any(x in t for x in ("Formal-results placeholder", "AUTO-GENERATED RESULT STATUS", "calibration_not_certified")) and (_ for _ in ()).throw(AssertionError(str(p)+": 残留旧稿状态或占位符"))))(p.read_text(encoding="utf-8"),p) for p in files]; print("静态检查通过：中英文稿环境、括号、引用和旧稿残留均正常。")'

draft-pdf: figures check
	@mkdir -p "$(OUTPUT_DIR)" "$(ZH_OUTPUT_DIR)"
	$(TECTONIC) --keep-logs --keep-intermediates --outdir "$(OUTPUT_DIR)" "$(ENGLISH_MAIN)"
	$(TECTONIC) --keep-logs --keep-intermediates --outdir "$(ZH_OUTPUT_DIR)" "$(CHINESE_MAIN)"
	@python3 -c 'import pathlib,sys; logs=[pathlib.Path(x) for x in sys.argv[1:]]; assert all("Overfull \\hbox" not in p.read_text(encoding="utf-8") for p in logs), "PDF 日志存在 Overfull hbox"; print("中英文稿编译通过：无 Overfull hbox。")' "$(OUTPUT_DIR)/main-en.log" "$(ZH_OUTPUT_DIR)/main.log"

final-pdf: figures check
	@mkdir -p "$(OUTPUT_DIR)"
	$(TECTONIC) --keep-logs --keep-intermediates --outdir "$(OUTPUT_DIR)" "$(ENGLISH_MAIN)"
	@python3 -c 'import pathlib,shutil,sys; log=pathlib.Path(sys.argv[1]); assert "Overfull \\hbox" not in log.read_text(encoding="utf-8"), "英文稿存在 Overfull hbox"; shutil.copyfile(sys.argv[2],sys.argv[3]); print("英文 arXiv PDF 编译通过。")' "$(OUTPUT_DIR)/main-en.log" "$(OUTPUT_DIR)/main-en.pdf" "$(OUTPUT_DIR)/main.pdf"

final-pdf-zh: figures check
	@mkdir -p "$(ZH_OUTPUT_DIR)"
	$(TECTONIC) --keep-logs --keep-intermediates --outdir "$(ZH_OUTPUT_DIR)" "$(CHINESE_MAIN)"
	@python3 -c 'import pathlib,sys; log=pathlib.Path(sys.argv[1]); assert "Overfull \\hbox" not in log.read_text(encoding="utf-8"), "中文稿存在 Overfull hbox"; print("中文复核 PDF 编译通过。")' "$(ZH_OUTPUT_DIR)/main.log"

check-isolated: figures check
	@set -eu; \
		tmp=$$(mktemp -d "$${TMPDIR:-/tmp}/kstar-arxiv.XXXXXX"); \
		src="$$tmp/source"; out="$$tmp/output"; \
		mkdir -p "$$src/figures" "$$out/en" "$$out/zh"; \
		cp Makefile README.md "$(ENGLISH_MAIN)" "$(CHINESE_MAIN)" references.bib "$$src/"; \
		cp figures/stopping-evaluation-framework.pdf figures/h0-stopping-headroom.pdf figures/confirmatory-quality-cost-risk.pdf "$$src/figures/"; \
		$(MAKE) -C "$$src" check; \
		$(TECTONIC) --keep-logs --keep-intermediates --outdir "$$out/en" "$$src/$(ENGLISH_MAIN)"; \
		$(TECTONIC) --keep-logs --keep-intermediates --outdir "$$out/zh" "$$src/$(CHINESE_MAIN)"; \
		python3 -c 'import pathlib,sys; root=pathlib.Path(sys.argv[1]); files=[p for p in root.rglob("*") if p.is_file() and p.suffix in {".tex",".bib"}]; leaks=[str(p.relative_to(root)) for p in files if any(x in p.read_text(encoding="utf-8") for x in ("/Users/","/root/","/workspace/"))]; assert not leaks, "稿件包含绝对路径: "+str(leaks)' "$$src"; \
		test -s "$$out/en/main-en.pdf"; test -s "$$out/zh/main.pdf"; \
		echo "隔离构建通过：$$tmp"

clean:
	$(RM) main.aux main.bbl main.blg main.log main.out main.pdf main.toc
	$(RM) main-en.aux main-en.bbl main-en.blg main-en.log main-en.out main-en.pdf main-en.toc
