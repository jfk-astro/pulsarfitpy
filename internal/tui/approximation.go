package tui

import (
	"fmt"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/jfk-astro/pulsarfitpy/internal/runner"
)

type approximationResultMsg struct {
	result string
	err    error
}

func (m Model) updateApproximation(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			m.state = menuView
			m.inputs = nil
			return m, nil

		case "tab", "shift+tab", "up", "down":
			if msg.String() == "up" || msg.String() == "shift+tab" {
				m.activeInput--
			} else {
				m.activeInput++
			}

			if m.activeInput < 0 {
				m.activeInput = len(m.inputs) - 1
			} else if m.activeInput >= len(m.inputs) {
				m.activeInput = 0
			}

			for i := range m.inputs {
				if i == m.activeInput {
					m.inputs[i].Focus()
				} else {
					m.inputs[i].Blur()
				}
			}

			cmds := make([]tea.Cmd, len(m.inputs))
			for i := range m.inputs {
				cmds[i] = m.inputs[i].Cursor.BlinkCmd()
			}
			return m, tea.Batch(cmds...)

		case "enter":
			return m, m.runApproximation()

		default:
			cmd := m.updateInputs(msg)
			return m, cmd
		}

	case approximationResultMsg:
		m.results = msg.result
		m.err = msg.err
		if msg.err == nil {
			m.state = resultsView
		}

		return m, nil
	}

	return m, nil
}

func (m *Model) updateInputs(msg tea.Msg) tea.Cmd {
	cmds := make([]tea.Cmd, len(m.inputs))

	for i := range m.inputs {
		m.inputs[i], cmds[i] = m.inputs[i].Update(msg)
	}

	return tea.Batch(cmds...)
}

func (m Model) runApproximation() tea.Cmd {
	return func() tea.Msg {
		xParam := m.inputs[0].Value()
		if xParam == "" {
			xParam = "P0"
		}

		yParam := m.inputs[1].Value()
		if yParam == "" {
			yParam = "P1"
		}

		degreeStr := m.inputs[2].Value()
		if degreeStr == "" {
			degreeStr = "5"
		}
		degree, err := strconv.Atoi(degreeStr)
		if err != nil {
			return approximationResultMsg{err: fmt.Errorf("invalid degree: %v", err)}
		}

		logScale := strings.ToLower(m.inputs[3].Value())
		logX := logScale == "y" || logScale == "yes" || logScale == ""
		logY := logX

		result, err := runner.RunApproximation(xParam, yParam, degree, logX, logY)
		if err != nil {
			return approximationResultMsg{err: err}
		}

		return approximationResultMsg{result: result}
	}
}
