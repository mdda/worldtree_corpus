WORLDTREE := tg2021-alldata-practice.zip

dataset: $(WORLDTREE)
	unzip -o '$<'

SHA256 := $(if $(shell which sha256sum),sha256sum,shasum -a 256)

$(WORLDTREE): tg2021task-practice.sha256
	@echo 'Please note that this distribution is subject to the terms set in the license:'
	@echo 'http://cognitiveai.org/explanationbank/'
	curl -sL -o '$@' 'http://cognitiveai.org/dist/$(WORLDTREE)'
	$(SHA256) -c "$<"
