package runner

import (
	"bytes"
	"fmt"
	"os/exec"
	"path/filepath"
	"runtime"
)

func RunApproximation(xParam, yParam string, degree int, logX, logY bool) (string, error) {
	pythonCmd := getPythonCommand()
	scriptPath := getScriptPath("headless.py")

	args := []string{
		scriptPath,
		"approximate",
		"--x-param", xParam,
		"--y-param", yParam,
		"--degree", fmt.Sprintf("%d", degree),
	}

	if logX {
		args = append(args, "--log-x")
	}
	if logY {
		args = append(args, "--log-y")
	}

	return executePython(pythonCmd, args...)
}

func RunPINN(xParam, yParam string, epochs int) (string, error) {
	pythonCmd := getPythonCommand()
	scriptPath := getScriptPath("headless.py")

	args := []string{
		scriptPath,
		"pinn",
		"--x-param", xParam,
		"--y-param", yParam,
		"--epochs", fmt.Sprintf("%d", epochs),
	}

	return executePython(pythonCmd, args...)
}

func RunQuery(params []string) (string, error) {
	pythonCmd := getPythonCommand()
	scriptPath := getScriptPath("headless.py")

	args := []string{scriptPath, "query"}
	for _, p := range params {
		args = append(args, "--param", p)
	}

	return executePython(pythonCmd, args...)
}

func getPythonCommand() string {
	if runtime.GOOS == "windows" {
		return "python"
	}

	return "python3"
}

func getScriptPath(filename string) string {
	_, file, _, _ := runtime.Caller(0)
	projectRoot := filepath.Join(filepath.Dir(file), "..", "..")

	return filepath.Join(projectRoot, "src", "pulsarfitpy", "pulsarfitpy", filename)
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
