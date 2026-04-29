package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strings"
)

// SECTION_START error_helpers

func WrapN(err error, callerSkip int) error {
	if err == nil {
		return nil
	}
	pc, _, line, ok := runtime.Caller(callerSkip)
	if !ok {
		return fmt.Errorf("wrap: error calling runtime.Caller")
	}
	f := runtime.FuncForPC(pc)
	return fmt.Errorf("%s:%d: %w", f.Name(), line, err)
}

func Wrap(err error) error {
	return WrapN(err, 2)
}

func WrapMsg(mfmt string, args ...any) error {
	if mfmt == "" {
		return nil
	}
	pc, _, line, ok := runtime.Caller(1)
	if !ok {
		return fmt.Errorf("wrap: error calling runtime.Caller")
	}
	f := runtime.FuncForPC(pc)
	return fmt.Errorf(
		"%s:%d: %s", f.Name(), line, fmt.Errorf(mfmt, args...),
	)
}

func fck(err error) {
	if err == nil {
		return
	}
	panic(err)
}

// SECTION_START main

type llmMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type llmCtx struct {
	// config
	apiEndpoint string
	apiKey      string
	model       string
	// If provided, execute prompt, print result and exit
	prompt string

	// state
	messages []llmMessage
}

type messageOutput struct {
	messageFrame string
	err          error
}

func llmStream(c *llmCtx, out chan<- messageOutput) {
	reqData := struct {
		Model    string       `json:"model"`
		Messages []llmMessage `json:"messages"`
		Stream   bool         `json:"stream"`
	}{c.model, c.messages, true}
	data, err := json.Marshal(reqData)
	if err != nil {
		out <- messageOutput{err: Wrap(err)}
		return
	}
	req, err := http.NewRequest(
		"POST", c.apiEndpoint, bytes.NewReader(data),
	)
	if err != nil {
		out <- messageOutput{err: Wrap(err)}
		return
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	hc := http.Client{}
	res, err := hc.Do(req)
	if err != nil {
		out <- messageOutput{err: Wrap(err)}
		return
	}
	if s := res.StatusCode; s != 200 {
		out <- messageOutput{err: WrapMsg("bad status code %d", s)}
		return
	}
	defer res.Body.Close()
	buf := bufio.NewScanner(res.Body)
	var fullMsg bytes.Buffer
	type resultFrame struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	for buf.Scan() {
		line := buf.Text()
		_, ok := strings.CutPrefix(line, "data: [DONE]")
		if ok {
			continue
		}
		mFrame, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var f resultFrame
		mFrameBytes := []byte(mFrame)
		err := json.Unmarshal(mFrameBytes, &f)
		if err != nil {
			out <- messageOutput{
				err: WrapMsg("%w: %s", err, mFrameBytes),
			}
			return
		}
		for _, c := range f.Choices {
			pMsg := c.Delta.Content
			if pMsg == "" {
				continue
			}
			fullMsg.WriteString(pMsg)
			out <- messageOutput{
				messageFrame: pMsg,
			}
		}
	}
	if fullMsg.Len() != 0 {
		c.messages = append(c.messages, llmMessage{
			Role:    "assistant",
			Content: fullMsg.String(),
		})
	}
}

func llmStreamStart(c *llmCtx) <-chan messageOutput {
	s := make(chan messageOutput, 1)
	go func() {
		llmStream(c, s)
		close(s)
	}()
	return s
}

type llmConfigError []string

func (l *llmConfigError) Error() string {
	var buf bytes.Buffer
	for _, s := range []string(*l) {
		buf.WriteString("\n\t")
		buf.WriteString(s)
	}
	buf.WriteString("\n")
	return buf.String()
}

func llmConfigParse(c *llmCtx, args []string) error {
	cfgTable := []struct {
		flagName     string
		envName      string
		defaultValue string
		dest         any
		required     bool
		desc string
	}{
		{
			"api-endpoint",
			"SLM_API_ENDPOINT",
			"",
			&c.apiEndpoint,
			true,
			"the chat completion API endpoint. ex: https://example.com/api/v1/chat/completions",
		},
		{
			"api-key",
			"SLM_API_KEY",
			"",
			&c.apiKey,
			true,
			"the API key to authenticate with the LLM provider",
		},
		{
			"model",
			"SLM_MODEL",
			"",
			&c.model,
			true,
			"the name of the model to be used (check your LLM API provider)",
		},
		{
			"p",
			"",
			"",
			&c.prompt,
			false,
			"a prompt to be executed",
		},
	}
	fSet := flag.NewFlagSet("slm", flag.ExitOnError)
	for _, cfg := range cfgTable {
		if cfg.flagName == "" {
			continue
		}
		switch dest := cfg.dest.(type) {
		case *string:
			fSet.StringVar(dest, cfg.flagName, cfg.defaultValue, cfg.desc)
		default:
			return WrapMsg("bad type: %T", dest)
		}
	}
	err := fSet.Parse(args)
	if err != nil {
		return Wrap(err)
	}
	for _, cfg := range cfgTable {
		if cfg.envName == "" {
			continue
		}
		env, ok := os.LookupEnv(cfg.envName)
		if !ok {
			continue
		}
		switch dest := cfg.dest.(type) {
		case *string:
			if *dest != "" {
				continue
			}
			*dest = env
		default:
			return WrapMsg("bad type: %T", dest)
		}
	}
	var errSlice []string
	for _, cfg := range cfgTable {
		if cfg.required == false {
			continue
		}
		switch dest := cfg.dest.(type) {
		case *string:
			if *dest != "" {
				continue
			}
			errSlice = append(
				errSlice,
				fmt.Sprintf(
					"undefined env %s or flag --%s",
					cfg.envName, cfg.flagName,
				),
			)
		default:
			return WrapMsg("bad type: %T", dest)
		}
	}
	if len(errSlice) != 0 {
		ce := llmConfigError(errSlice)
		return WrapMsg("error: %w", &ce)
	}
	return nil
}

func consoleLoop() error {
	var (
		lc llmCtx
		data string
		ok bool
		history []llmMessage
		stop bool
	)
	err := llmConfigParse(&lc, os.Args[1:])
	if err != nil {
		return Wrap(err)
	}

	status := func() string {
		return lc.model
	}

	if lc.prompt != "" {
		lc.messages = append(lc.messages, llmMessage{
			Role:    "user",
			Content: lc.prompt,
		})
		output := llmStreamStart(&lc)
		for m := range output {
			if m.err != nil {
				fmt.Println(
					WrapMsg("error: %w", m.err),
				)
				return nil
			}
			fmt.Print(m.messageFrame)
		}
		fmt.Print("\n")
		return nil
	}

	s := bufio.NewScanner(os.Stdin)
	clearScreen := func() {
		fmt.Printf("\x1b[2J" + "\x1b[0;0H")
	}
	slmMessage := func() {
		fmt.Printf("SLM by allocz\n")
	}
	prompt := func() {
		fmt.Printf(
			"\x1b[0;36m" +
			"(%s)>>>" +
			"\x1b[0m" +
			" ",
			status(),
		)
	}
	clearScreen()
	slmMessage()
	prompt()
	for !stop && s.Scan() {
		func() {
			defer func() {
				if stop {
					return
				}
				prompt()
			}()
			line := s.Text()
			data, ok = strings.CutPrefix(line, "/echo ")
			if ok {
				fmt.Printf("%s\n", data)
				return
			}
			data, ok = strings.CutPrefix(line, "/clear")
			if ok {
				clearScreen()
				history = append(history, lc.messages...)
				lc.messages = lc.messages[:0]
				return
			}
			data, ok = strings.CutPrefix(line, "/history")
			if ok {
				for _, m := range history {
					if m.Role == "user" {
						fmt.Print(">>> ")
					}
					fmt.Printf("%s\n\n", m.Content)
				}
				return
			}
			data, ok = strings.CutPrefix(line, "/q")
			if ok {
				stop = true
				return
			}
			data, ok = strings.CutPrefix(line, "/help")
			if ok {
				fmt.Println("TODO")
				return
			}
			data, ok = strings.CutPrefix(line, "/")
			if ok {
				fmt.Println("Unknown command")
				return
			}
			lc.messages = append(lc.messages, llmMessage{
				Role:    "user",
				Content: line,
			})
			output := llmStreamStart(&lc)
			fmt.Print("\n")
			for m := range output {
				if m.err != nil {
					fmt.Println(WrapMsg("error: %w", m.err))
					return
				}
				fmt.Print(m.messageFrame)
			}
			fmt.Print("\n\n")
		}()
	}
	return nil
}

func run() error {
	return consoleLoop()
}

func main() {
	err := run()
	if err != nil {
		log.Fatal(err)
	}
}
