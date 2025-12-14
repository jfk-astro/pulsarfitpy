.PHONY: build clean deps run

build: deps
	go build -o bin/pulsar-cli.exe ./cmd/pulsar-cli

deps:
	go mod download
	go mod tidy

run: build
	.\bin\pulsar-cli.exe

clean:
	@if exist bin rmdir /s /q bin
