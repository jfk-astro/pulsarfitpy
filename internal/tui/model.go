package tui

import (
	"fmt"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type sessionState int

const (
	menuView sessionState = iota
	approximationView
	pinnView
	queryView
	resultsView
)

type Model struct {
	state       sessionState
	menu        list.Model
	inputs      []textinput.Model
	results     string
	err         error
	width       int
	height      int
	activeInput int
}

type item struct {
	title       string
	description string
}

func (i item) Title() string       { return i.title }
func (i item) Description() string { return i.description }
func (i item) FilterValue() string { return i.title }

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("205")).
			MarginLeft(2)

	helpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("241")).
			MarginLeft(2)

	docStyle = lipgloss.NewStyle().Margin(1, 2)
)

func NewModel() Model {
	items := []list.Item{
		item{
			title:       "Polynomial Approximation",
			description: "Fit polynomial models to pulsar data",
		},
		item{
			title:       "Physics-Informed Neural Network",
			description: "Train PINN to learn physical constants",
		},
		item{
			title:       "Query ATNF Database",
			description: "Query pulsar catalogue parameters",
		},
		item{
			title:       "Exit",
			description: "Quit the application",
		},
	}

	delegate := list.NewDefaultDelegate()
	menuList := list.New(items, delegate, 0, 0)
	menuList.Title = "pulsarfitpy - Pulsar Analysis TUI"
	menuList.SetShowStatusBar(false)
	menuList.SetFilteringEnabled(false)

	return Model{
		state: menuView,
		menu:  menuList,
	}
}

func (m Model) Init() tea.Cmd {
	return nil
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			if m.state == menuView {
				return m, tea.Quit
			}

			m.state = menuView
			m.results = ""
			m.err = nil

			return m, nil
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.menu.SetSize(msg.Width-4, msg.Height-4)

		return m, nil
	}

	return m.updateBasedOnState(msg)
}

func (m Model) updateBasedOnState(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch m.state {
	case menuView:
		return m.updateMenu(msg)
	case approximationView:
		return m.updateApproximation(msg)
	case pinnView:
		return m.updatePINN(msg)
	case queryView:
		return m.updateQuery(msg)
	case resultsView:
		return m.updateResults(msg)
	}

	return m, cmd
}

func (m Model) View() string {
	switch m.state {
	case menuView:
		return m.viewMenu()
	case approximationView:
		return m.viewApproximation()
	case pinnView:
		return m.viewPINN()
	case queryView:
		return m.viewQuery()
	case resultsView:
		return m.viewResults()
	}

	return ""
}

func (m Model) viewMenu() string {
	return docStyle.Render(m.menu.View())
}

func (m Model) updateMenu(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "enter":
			selectedItem := m.menu.SelectedItem()
			if selectedItem != nil {
				choice := selectedItem.(item)
				switch choice.title {
				case "Polynomial Approximation":
					m.state = approximationView
					m.inputs = makeApproximationInputs()

					return m, textinput.Blink
				case "Physics-Informed Neural Network":
					m.state = pinnView
					m.inputs = makePINNInputs()

					return m, textinput.Blink
				case "Query ATNF Database":
					m.state = queryView
					m.inputs = makeQueryInputs()

					return m, textinput.Blink
				case "Exit":
					return m, tea.Quit
				}
			}
		}
	}

	var cmd tea.Cmd
	m.menu, cmd = m.menu.Update(msg)

	return m, cmd
}

func makeApproximationInputs() []textinput.Model {
	inputs := make([]textinput.Model, 4)

	inputs[0] = textinput.New()
	inputs[0].Placeholder = "P0"
	inputs[0].Focus()
	inputs[0].CharLimit = 20
	inputs[0].Width = 30
	inputs[0].Prompt = "X Parameter: "

	inputs[1] = textinput.New()
	inputs[1].Placeholder = "P1"
	inputs[1].CharLimit = 20
	inputs[1].Width = 30
	inputs[1].Prompt = "Y Parameter: "

	inputs[2] = textinput.New()
	inputs[2].Placeholder = "5"
	inputs[2].CharLimit = 2
	inputs[2].Width = 30
	inputs[2].Prompt = "Max Degree:  "

	inputs[3] = textinput.New()
	inputs[3].Placeholder = "y"
	inputs[3].CharLimit = 1
	inputs[3].Width = 30
	inputs[3].Prompt = "Log Scale (y/n): "

	return inputs
}

func makePINNInputs() []textinput.Model {
	inputs := make([]textinput.Model, 3)

	inputs[0] = textinput.New()
	inputs[0].Placeholder = "P0"
	inputs[0].Focus()
	inputs[0].CharLimit = 20
	inputs[0].Width = 30
	inputs[0].Prompt = "X Parameter: "

	inputs[1] = textinput.New()
	inputs[1].Placeholder = "P1"
	inputs[1].CharLimit = 20
	inputs[1].Width = 30
	inputs[1].Prompt = "Y Parameter: "

	inputs[2] = textinput.New()
	inputs[2].Placeholder = "3000"
	inputs[2].CharLimit = 6
	inputs[2].Width = 30
	inputs[2].Prompt = "Epochs:      "

	return inputs
}

func makeQueryInputs() []textinput.Model {
	inputs := make([]textinput.Model, 1)

	inputs[0] = textinput.New()
	inputs[0].Placeholder = "P0,P1,DM"
	inputs[0].Focus()
	inputs[0].CharLimit = 100
	inputs[0].Width = 50
	inputs[0].Prompt = "Parameters (comma-separated): "

	return inputs
}

func (m Model) viewApproximation() string {
	var s string
	s += titleStyle.Render("Polynomial Approximation") + "\n\n"

	for i := range m.inputs {
		s += m.inputs[i].View() + "\n"
	}

	s += "\n" + helpStyle.Render("↑/↓: navigate • enter: submit • esc: back to menu")

	if m.err != nil {
		s += "\n\n" + lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Render(fmt.Sprintf("Error: %v", m.err))
	}

	return docStyle.Render(s)
}

func (m Model) viewPINN() string {
	var s string
	s += titleStyle.Render("Physics-Informed Neural Network") + "\n\n"

	for i := range m.inputs {
		s += m.inputs[i].View() + "\n"
	}

	s += "\n" + helpStyle.Render("↑/↓: navigate • enter: submit • esc: back to menu")

	if m.err != nil {
		s += "\n\n" + lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Render(fmt.Sprintf("Error: %v", m.err))
	}

	return docStyle.Render(s)
}

func (m Model) viewQuery() string {
	var s string
	s += titleStyle.Render("Query ATNF Database") + "\n\n"

	for i := range m.inputs {
		s += m.inputs[i].View() + "\n"
	}

	s += "\n" + helpStyle.Render("enter: submit • esc: back to menu")

	if m.err != nil {
		s += "\n\n" + lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Render(fmt.Sprintf("Error: %v", m.err))
	}

	return docStyle.Render(s)
}

func (m Model) viewResults() string {
	var s string
	s += titleStyle.Render("Results") + "\n\n"
	s += m.results + "\n\n"
	s += helpStyle.Render("esc: back to menu • q: quit")

	return docStyle.Render(s)
}
