build:
	python setup.py bdist_wheel

clean:
	rm -r build dist Fern2.egg-info

upload:
	which python
	python -m twine upload dist/*
	make clean
