/* Copy-only client helpers. No requests, model calls, installation or MCP execution. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const prompt = $('agent-prompt');

  if (location.protocol === 'http:' || location.protocol === 'https:') {
    const origin = location.origin;
    prompt.value = `Connect to Interaction Twin: ${origin}/interaction_twin/showcase/agent-setup.html`;
  }

  function wireCopy(buttonId, field, statusId) {
    const button = $(buttonId);
    const status = $(statusId);
    function feedback(copied) {
      const unchanged = field.value === copied;
      button.setAttribute('data-copy-state', unchanged ? 'copied' : 'idle');
      status.textContent = unchanged ? 'Copied.' : 'The field changed. Copy the updated text.';
    }
    button.addEventListener('click', async () => {
      if (button.disabled) return;
      const copied = field.value;
      button.setAttribute('data-copy-state', 'idle');
      status.textContent = '';
      try {
        if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(copied);
        // A path edited while a clipboard request is pending must not look copied.
        feedback(copied);
      } catch (_) {
        const selected = field.value;
        field.focus();
        field.select();
        // Older/insecure-context browsers can still copy from this explicit
        // click. If they reject that too, keep the line selected for ⌘/Ctrl+C.
        try {
          if (typeof document.execCommand === 'function' && document.execCommand('copy')) {
            feedback(selected);
            return;
          }
        } catch (_) { /* Selection remains a usable keyboard fallback. */ }
        status.textContent = 'Text selected. Press ⌘C on Mac or Ctrl+C to copy.';
      }
    });
  }

  wireCopy('copy-agent-prompt', prompt, 'prompt-copy-status');
  wireCopy('copy-charge-prompt', $('charge-example-prompt'), 'charge-copy-status');
})();
