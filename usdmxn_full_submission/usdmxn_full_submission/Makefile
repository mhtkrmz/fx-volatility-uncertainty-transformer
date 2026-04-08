.PHONY: all prepare baselines transformer tables figures paper clean

all:
	python run_all.py

prepare:
	python code/01_prepare_data.py

baselines:
	python code/02_fit_baselines.py

transformer:
	python code/03_train_transformer.py

tables:
	python code/04_build_tables.py

figures:
	python code/05_make_figures.py

paper:
	python code/06_build_paper.py

clean:
	rm -f outputs/processed/*.csv outputs/processed/*.gz outputs/models/*.csv outputs/models/*.json \
	      outputs/predictions/*.csv outputs/tables/*.csv outputs/tables/*.tex \
	      outputs/reports/*.json outputs/reports/*.txt outputs/figures/*.pdf outputs/figures/*.png
