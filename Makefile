.PHONY: build clean

build:
	go build -o bin/pulsar-cli.exe ./cmd/pulsar-cli

clean:
	@if exist bin rmdir /s /q bin
