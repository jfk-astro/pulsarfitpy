package tui

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/jfk-astro/pulsarfitpy/internal/runner"
)

type queryResultMsg struct {
	result string
	err    error
}

func (m Model) updateQuery(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			m.state = menuView
			m.inputs = nil

			return m, nil

		case "enter":
			return m, m.runQuery()

		default:
			cmd := m.updateInputs(msg)
			return m, cmd
		}

	case queryResultMsg:
		m.results = msg.result
		m.err = msg.err

		if msg.err == nil {
			m.state = resultsView
		}

		return m, nil
	}

	return m, nil
}

func (m Model) runQuery() tea.Cmd {
	return func() tea.Msg {
		paramsStr := m.inputs[0].Value()

		if paramsStr == "" {
			paramsStr = "P0,P1"
		}

		params := strings.Split(paramsStr, ",")
		for i := range params {
			params[i] = strings.TrimSpace(params[i])
		}

		result, err := runner.RunQuery(params)
		if err != nil {
			return queryResultMsg{err: err}
		}

		return queryResultMsg{result: result}
	}
}
