package tui

import (
	tea "github.com/charmbracelet/bubbletea"
)

func (m Model) updateResults(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc", "q":
			m.state = menuView
			m.results = ""
			m.err = nil

			return m, nil
		}
	}

	return m, nil
}
