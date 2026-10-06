(function () {
    const API_URL = "https://baillietn-hilbertlm-space.hf.space/chat";
    // Sending only the 3 last messages...
    const CONTEXT_MESSAGES = 3;

    const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const coarsePointer = matchMedia('(pointer: coarse)').matches;

    const app = document.getElementById('app');
    const thread = document.getElementById('thread');
    const messagesEl = document.getElementById('messages');
    const composer = document.getElementById('composer');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const newChatButton = document.getElementById('new-chat');
    const toBottomButton = document.getElementById('to-bottom');
    const suggestions = document.getElementById('suggestions');
    const greetingEl = document.getElementById('greeting');

    const svg = (body) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
    const ICONS = {
        copy: svg('<rect x="9" y="9" width="13" height="13" rx="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>'),
        check: svg('<polyline points="20 6 9 17 4 12"></polyline>'),
        retry: svg('<polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>')
    };

    let conversationHistory = [];
    let turns = [];
    let currentAbortController = null;
    let isGenerating = false;

    // ---------- markdown ----------

    const MATH_BLOCKS = /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)/g;
    const KATEX_OPTIONS = {
        delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true }
        ],
        throwOnError: false
    };

    function escapeHtml(text) {
        return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function decorateCodeBlock(pre) {
        const code = pre.querySelector('code');
        if (!code) return;

        const lang = (code.className.match(/language-([\w+#-]+)/) || [])[1];
        if (lang && window.hljs && hljs.getLanguage(lang)) hljs.highlightElement(code);

        const head = document.createElement('div');
        head.className = 'code-head';
        const label = document.createElement('span');
        label.textContent = lang || 'text';
        const copy = document.createElement('button');
        copy.type = 'button';
        copy.className = 'code-copy';
        copy.innerHTML = `${ICONS.copy}<span>Copy</span>`;
        head.append(label, copy);

        const wrapper = document.createElement('div');
        wrapper.className = 'code';
        pre.replaceWith(wrapper);
        wrapper.append(head, pre);
    }

    function renderMarkdown(target, text) {
        if (!window.marked || !window.DOMPurify) {
            target.textContent = text;
            return;
        }

        // Keep LaTeX away from the markdown parser, then hand it back to KaTeX untouched.
        const mathBlocks = [];
        const protectedText = text.replace(MATH_BLOCKS, (match) => {
            mathBlocks.push(match);
            return `MATHBLOCKPLACEHOLDER${mathBlocks.length - 1}END`;
        });

        const parsedHtml = marked.parse(protectedText, { breaks: true })
            .replace(/MATHBLOCKPLACEHOLDER(\d+)END/g, (_, index) => escapeHtml(mathBlocks[index]));

        target.innerHTML = DOMPurify.sanitize(parsedHtml);

        target.querySelectorAll('a').forEach((link) => {
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
        });
        target.querySelectorAll('pre').forEach(decorateCodeBlock);

        if (window.renderMathInElement) {
            try {
                renderMathInElement(target, KATEX_OPTIONS);
            } catch (e) {
                console.error("Erreur de rendu KaTeX :", e);
            }
        }
    }

    // ---------- scrolling ----------

    // `stick` follows the stream while the reader stays at the bottom;
    // `seeking` covers the smooth scroll we trigger ourselves.
    let stick = true;
    let seeking = false;
    let seekTimer = null;

    function isAtBottom() {
        return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 48;
    }

    function followStream() {
        if (stick) thread.scrollTop = thread.scrollHeight;
        else toBottomButton.classList.add('is-visible');
    }

    function scrollToBottom(smooth) {
        stick = true;
        toBottomButton.classList.remove('is-visible');
        if (isAtBottom() || !smooth || reduceMotion) {
            thread.scrollTop = thread.scrollHeight;
            return;
        }
        seeking = true;
        clearTimeout(seekTimer);
        seekTimer = setTimeout(() => { seeking = false; }, 800);
        thread.scrollTo({ top: thread.scrollHeight, behavior: 'smooth' });
    }

    thread.addEventListener('scroll', () => {
        const bottom = isAtBottom();
        if (seeking && !bottom) return;
        seeking = false;
        stick = bottom;
        toBottomButton.classList.toggle('is-visible', !bottom);
    }, { passive: true });

    ['wheel', 'touchmove'].forEach((type) => {
        thread.addEventListener(type, () => { seeking = false; }, { passive: true });
    });

    addEventListener('resize', () => {
        if (app.dataset.view === 'chat') followStream();
    });

    toBottomButton.addEventListener('click', () => scrollToBottom(true));

    // ---------- composer ----------

    function autosizeInput() {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    }

    function syncComposer() {
        sendButton.dataset.mode = isGenerating ? 'stop' : 'send';
        sendButton.disabled = !isGenerating && !userInput.value.trim();
        sendButton.setAttribute('aria-label', isGenerating ? 'Stop response' : 'Send message');
    }

    function setInput(value) {
        userInput.value = value;
        autosizeInput();
        syncComposer();
    }

    // Switches between the centered home composer and the docked chat composer,
    // sliding the box from its old position to the new one.
    function setView(view) {
        if (app.dataset.view === view) return;

        const before = composer.getBoundingClientRect();
        app.dataset.view = view;
        userInput.placeholder = view === 'home' ? 'How can I help you today?' : 'Reply to HilbertLM...';
        autosizeInput();
        const after = composer.getBoundingClientRect();

        if (reduceMotion || !composer.animate) return;
        composer.animate(
            [{ transform: `translateY(${before.top - after.top}px)` }, { transform: 'none' }],
            { duration: 480, easing: 'cubic-bezier(.2, .7, .2, 1)' }
        );
        if (view === 'chat') {
            thread.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 360, easing: 'ease-out' });
        }
    }

    // ---------- turns ----------

    function createTurn(prompt) {
        const el = document.createElement('section');
        el.className = 'turn';

        const userDiv = document.createElement('div');
        userDiv.className = 'msg-user';
        userDiv.textContent = prompt;

        const assistantDiv = document.createElement('div');
        assistantDiv.className = 'msg-assistant';
        const body = document.createElement('div');
        body.className = 'prose';
        const foot = document.createElement('div');
        foot.className = 'msg-foot';
        assistantDiv.append(body, foot);

        el.append(userDiv, assistantDiv);
        messagesEl.appendChild(el);

        const turn = { el, body, foot, prompt, text: '', committed: false };
        turns.push(turn);
        return turn;
    }

    function removeTurn(turn) {
        turn.el.remove();
        turns = turns.filter((t) => t !== turn);
    }

    function showActions(turn) {
        turn.foot.innerHTML =
            `<button type="button" class="act-btn act-copy" title="Copy" aria-label="Copy response">${ICONS.copy}</button>` +
            `<button type="button" class="act-btn act-retry" title="Retry" aria-label="Retry response">${ICONS.retry}</button>`;
    }

    function showError(turn, message) {
        turn.body.innerHTML = '';
        turn.foot.innerHTML = '';

        const notice = document.createElement('div');
        notice.className = 'notice';
        const label = document.createElement('span');
        label.textContent = message;
        const retry = document.createElement('button');
        retry.type = 'button';
        retry.className = 'notice-retry';
        retry.textContent = 'Retry';
        notice.append(label, retry);
        turn.body.appendChild(notice);
    }

    async function generate(turn) {
        const controller = new AbortController();
        currentAbortController = controller;
        isGenerating = true;
        syncComposer();

        turn.text = '';
        turn.body.innerHTML = '';
        turn.foot.innerHTML = '<span class="orb is-thinking" role="status" aria-label="Generating"></span>';

        const userMessage = { role: "user", content: turn.prompt };

        let paintQueued = false;
        const paint = () => {
            paintQueued = false;
            if (currentAbortController !== controller) return;
            renderMarkdown(turn.body, turn.text);
            followStream();
        };

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: [...conversationHistory, userMessage].slice(-CONTEXT_MESSAGES) }),
                signal: controller.signal
            });

            if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                turn.text += decoder.decode(value, { stream: true });
                if (!paintQueued) {
                    paintQueued = true;
                    requestAnimationFrame(paint);
                }
            }

            if (!turn.text.trim()) throw new Error('Empty response');
        } catch (error) {
            // The conversation was cleared while this request was in flight.
            if (!turns.includes(turn)) return;

            if (error.name !== 'AbortError') {
                console.error('API Error:', error);
                showError(turn, error.message === 'Empty response'
                    ? 'HilbertLM returned an empty response.'
                    : "Couldn't reach HilbertLM. The Hugging Face Space may be waking up, try again in a moment.");
                followStream();
                return;
            }

            console.log('Chatbot response was aborted...');
            if (!turn.text.trim()) {
                // Stopped before the first token: hand the prompt back, as if it was never sent.
                removeTurn(turn);
                if (!userInput.value.trim()) setInput(turn.prompt);
                if (turns.length === 0) setView('home');
                return;
            }
        } finally {
            if (currentAbortController === controller) {
                currentAbortController = null;
                isGenerating = false;
                syncComposer();
            }
        }

        // Completed, or stopped midway: either way the text on screen is the answer.
        renderMarkdown(turn.body, turn.text);
        showActions(turn);
        conversationHistory.push(userMessage, { role: "assistant", content: turn.text });
        turn.committed = true;
        followStream();
    }

    function handleSend(text) {
        const message = (text === undefined ? userInput.value : text).trim();
        if (!message || isGenerating) return;

        setView('chat');
        const turn = createTurn(message);
        setInput('');
        scrollToBottom(true);
        userInput.focus();
        generate(turn);
    }

    function retryTurn(turn) {
        if (isGenerating || turn !== turns[turns.length - 1]) return;

        if (turn.committed) {
            conversationHistory.splice(-2);
            turn.committed = false;
        }
        generate(turn);
    }

    function clearConversation() {
        const controller = currentAbortController;
        currentAbortController = null;
        isGenerating = false;
        if (controller) controller.abort();

        conversationHistory = [];
        turns = [];
        messagesEl.innerHTML = '';
        toBottomButton.classList.remove('is-visible');
        stick = true;

        setView('home');
        syncComposer();
        userInput.focus();

        console.log('Conversation history cleared');
    }

    // ---------- events ----------

    function flashCopied(button, restoreHtml, doneHtml) {
        button.innerHTML = doneHtml;
        button.classList.add('is-done');
        setTimeout(() => {
            button.innerHTML = restoreHtml;
            button.classList.remove('is-done');
        }, 1400);
    }

    async function copyText(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (e) {
            console.error('Copy failed:', e);
            return false;
        }
    }

    messagesEl.addEventListener('click', async (e) => {
        const button = e.target.closest('button');
        if (!button) return;

        if (button.classList.contains('code-copy')) {
            const code = button.closest('.code').querySelector('pre');
            if (await copyText(code.textContent)) {
                flashCopied(button, `${ICONS.copy}<span>Copy</span>`, `${ICONS.check}<span>Copied</span>`);
            }
            return;
        }

        const turnEl = button.closest('.turn');
        const turn = turns.find((t) => t.el === turnEl);
        if (!turn) return;

        if (button.classList.contains('act-copy')) {
            if (await copyText(turn.text)) flashCopied(button, ICONS.copy, ICONS.check);
        } else if (button.classList.contains('act-retry') || button.classList.contains('notice-retry')) {
            retryTurn(turn);
        }
    });

    sendButton.addEventListener('click', () => {
        if (isGenerating && currentAbortController) {
            currentAbortController.abort();
        } else {
            handleSend();
        }
    });

    composer.addEventListener('submit', (e) => {
        e.preventDefault();
        handleSend();
    });

    composer.addEventListener('click', (e) => {
        if (!e.target.closest('button')) userInput.focus();
    });

    userInput.addEventListener('input', () => {
        autosizeInput();
        syncComposer();
    });

    userInput.addEventListener('keydown', (e) => {
        // On touch keyboards Enter inserts a new line, the send button submits.
        if (e.key !== 'Enter' || e.shiftKey || e.isComposing || coarsePointer) return;
        e.preventDefault();
        handleSend();
    });

    suggestions.addEventListener('click', (e) => {
        const chip = e.target.closest('.chip');
        if (chip) handleSend(chip.dataset.prompt);
    });

    newChatButton.addEventListener('click', () => {
        if (isGenerating) {
            if (confirm('A message is being generated. Clear anyway?')) clearConversation();
        } else if (conversationHistory.length > 4) {
            if (confirm(`Clear ${conversationHistory.length} messages?`)) clearConversation();
        } else {
            clearConversation();
        }
    });

    const hour = new Date().getHours();
    greetingEl.textContent = hour >= 5 && hour < 12 ? 'Good morning'
        : hour >= 12 && hour < 18 ? 'Good afternoon'
            : 'Good evening';

    autosizeInput();
    syncComposer();
    userInput.focus();
})();

(function () {
    var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    var c = document.getElementById('stars'), x = c.getContext('2d');
    var w, h, dpr, stars = [], mx = 0, t = 0;
    function size() {
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        w = c.width = innerWidth * dpr; h = c.height = innerHeight * dpr;
        c.style.width = innerWidth + 'px'; c.style.height = innerHeight + 'px';
        var n = Math.round(innerWidth * innerHeight / 5600);
        stars = []; for (var i = 0; i < n; i++) stars.push({
            x: Math.random() * w, y: Math.random() * h, z: Math.random() * 0.8 + 0.2,
            r: (Math.random() * 1.3 + 0.3) * dpr, a: Math.random() * 0.5 + 0.35,
            tw: Math.random() * 1.6 + 0.4, ph: Math.random() * 6.28, hue: Math.random() < 0.25 ? 195 : 210
        });
    }
    addEventListener('resize', size);
    addEventListener('pointermove', function (e) { mx = (e.clientX / innerWidth - 0.5); });
    size();
    function frame() {
        t += 0.016; x.clearRect(0, 0, w, h);
        for (var i = 0; i < stars.length; i++) {
            var s = stars[i];
            if (!reduce) { s.y -= s.z * 0.16 * dpr; if (s.y < -2) s.y = h + 2; }
            var px = s.x + mx * 36 * s.z * dpr;
            var a = reduce ? s.a : s.a * (0.55 + 0.45 * Math.sin(t * s.tw + s.ph));
            x.beginPath(); x.fillStyle = 'hsla(' + s.hue + ',100%,82%,' + a + ')'; x.arc(px, s.y, s.r, 0, 6.283); x.fill();
            if (s.r > 1.4 * dpr) { x.beginPath(); x.fillStyle = 'hsla(' + s.hue + ',100%,80%,' + (a * 0.12) + ')'; x.arc(px, s.y, s.r * 3, 0, 6.283); x.fill(); }
        }
        requestAnimationFrame(frame);
    }
    if (reduce) { for (var i = 0; i < stars.length; i++) { var s = stars[i]; x.beginPath(); x.fillStyle = 'hsla(' + s.hue + ',100%,82%,' + s.a + ')'; x.arc(s.x, s.y, s.r, 0, 6.283); x.fill(); } }
    else requestAnimationFrame(frame);
})();
