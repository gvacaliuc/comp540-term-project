NOTEBOOKS = notebooks/*.ipynb
PYFILES = deton8/*.py
HTMLFILES = notebooks/*.html
ZIPFILE = rice_comp540_gv8_wvl1.zip
POSTERPDF = ./poster/gv8-wvl1-comp540-poster.pdf 
PAPERPDF = ./paper/gv8-wvl1-comp540-paper.pdf

all: zipfile

html:
	jupyter nbconvert --to html $(NOTEBOOKS) > /dev/null

zipfile: html
	zip -r $(ZIPFILE) \
		. \
		-i setup.py \
		-i README.md \
		-i $(PYFILES) \
		-i $(HTMLFILES) \
		-i $(NOTEBOOKS) \
		-i $(POSTERPDF) \
		-i $(PAPERPDF)

clean:
	/bin/rm -f $(HTMLFILES)
	/bin/rm -f $(ZIPFILE)

.PHONY: clean
