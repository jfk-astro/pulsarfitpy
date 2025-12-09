package runner

import (
	"bytes"
	_ "embed"
	"fmt"
	"os"
	"os/exec"
	"runtime"
)

//go:embed headless.py
var headlessPyScript []byte

func RunApproximation(xParam, yParam string, degree int, logX, logY bool) (string, error) {
	scriptPath, cleanup, err := writeEmbeddedScript()
	if err != nil {
		return "", fmt.Errorf("failed to create temp script: %w", err)
	}
	defer cleanup()

	pythonCmd := getPythonCommand()

	args := []string{
		scriptPath,
		"fit",
		"--json",
		fmt.Sprintf(`{"x_param":"%s","y_param":"%s","test_degree":%d,"log_x":%t,"log_y":%t}`,
			xParam, yParam, degree, logX, logY),
	}

	return executePython(pythonCmd, args...)
}

func RunPINN(xParam, yParam string, epochs int) (string, error) {
	scriptPath, cleanup, err := writeEmbeddedScript()
	if err != nil {
		return "", fmt.Errorf("failed to create temp script: %w", err)
	}
	defer cleanup()

	pythonCmd := getPythonCommand()

	args := []string{
		scriptPath,
		"fit",
		"--json",
		fmt.Sprintf(`{"x_param":"%s","y_param":"%s","test_degree":3,"epochs":%d}`,
			xParam, yParam, epochs),
	}

	return executePython(pythonCmd, args...)
}

func RunQuery(params []string) (string, error) {
	scriptPath, cleanup, err := writeEmbeddedScript()
	if err != nil {
		return "", fmt.Errorf("failed to create temp script: %w", err)
	}
	defer cleanup()

	pythonCmd := getPythonCommand()

	paramsJSON := "["
	for i, p := range params {
		if i > 0 {
			paramsJSON += ","
		}
		paramsJSON += fmt.Sprintf(`"%s"`, p)
	}
	paramsJSON += "]"

	args := []string{
		scriptPath,
		"query",
		"--json",
		fmt.Sprintf(`{"params":%s}`, paramsJSON),
	}

	return executePython(pythonCmd, args...)
}

func getPythonCommand() string {
	if runtime.GOOS == "windows" {
		return "python"
	}

	return "python3"
}

func writeEmbeddedScript() (string, func(), error) {
	tmpFile, err := os.CreateTemp("", "pulsarfitpy_*.py")
	if err != nil {
		return "", nil, err
	}

	if _, err := tmpFile.Write(headlessPyScript); err != nil {
		tmpFile.Close()
		os.Remove(tmpFile.Name())
		return "", nil, err
	}

	if err := tmpFile.Close(); err != nil {
		os.Remove(tmpFile.Name())
		return "", nil, err
	}

	cleanup := func() {
		os.Remove(tmpFile.Name())
	}

	return tmpFile.Name(), cleanup, nil
}

func executePython(pythonCmd string, args ...string) (string, error) {
	cmd := exec.Command(pythonCmd, args...)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("python execution failed: %v\nStderr: %s", err, stderr.String())
	}

	return stdout.String(), nil
}
