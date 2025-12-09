package tui

import (
	"fmt"
	"strconv"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/jfk-astro/pulsarfitpy/internal/runner"
)

type pinnResultMsg struct {
	result string
	err    error
}

func (m Model) updatePINN(msg tea.Msg) (tea.Model, tea.Cmd) {
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
			return m, m.runPINN()

		default:
			cmd := m.updateInputs(msg)
			return m, cmd
		}

	case pinnResultMsg:
		m.results = msg.result
		m.err = msg.err

		if msg.err == nil {
			m.state = resultsView
		}

		return m, nil
	}

	return m, nil
}

func (m Model) runPINN() tea.Cmd {
	return func() tea.Msg {
		xParam := m.inputs[0].Value()
		if xParam == "" {
			xParam = "P0"
		}

		yParam := m.inputs[1].Value()
		if yParam == "" {
			yParam = "P1"
		}

		epochsStr := m.inputs[2].Value()
		if epochsStr == "" {
			epochsStr = "3000"
		}

		epochs, err := strconv.Atoi(epochsStr)
		if err != nil {
			return pinnResultMsg{err: fmt.Errorf("invalid epochs: %v", err)}
		}

		result, err := runner.RunPINN(xParam, yParam, epochs)
		if err != nil {
			return pinnResultMsg{err: err}
		}

		return pinnResultMsg{result: result}
	}
}
