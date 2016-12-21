FMPATH=./libfm
BINPATH=$(FMPATH)/bin
SCRIPTPATH=$(FMPATH)/scripts

PY=python3

DIR=.
TEST=$(DIR)/test.txt
TEST_GEN=$(DIR)/test_fm.txt
TRAIN=$(DIR)/train.txt
TESTFM=$(DIR)/test_fm.txt.libfm
TRAINFM=$(DIR)/train.txt.libfm
OUTFM=$(DIR)/output.libfm

ITERN=1000

rowN=500
colN=1000

.PHONY: all clean

all: testgen convert mf

testgen:
	$(PY) generateTest.py $(TEST_GEN) $(rowN) $(colN)

convert: $(TEST_GEN)
	$(SCRIPTPATH)/triple_format_to_libfm.pl -in $(TRAIN),$(TEST_GEN) -target 2 -separator " "

mf: $(TRAINFM) $(TESTFM)
	$(BINPATH)/libFM -task r -train $(TRAINFM) -test $(TESTFM) -iter $(ITERN) -out $(OUTFM) -dim 1,1,8

clean:
