NOTEBOOKS = notebooks/*.ipynb
PYFILES = deton8/*.py
HTMLFILES = notebooks/*.html
ZIPFILE = rice_comp540_gv8_wvl1.zip

all: zipfile

html:
	jupyter nbconvert --to html $(NOTEBOOKS) > /dev/null

zipfile: html
	zip -r $(ZIPFILE) \
		. \
		-i setup.py \
		-i $(PYFILES) \
		-i $(HTMLFILES) \
		-i $(NOTEBOOKS)

clean:
	/bin/rm -f $(HTMLFILES)
	/bin/rm -f $(ZIPFILE)

.PHONY: clean
