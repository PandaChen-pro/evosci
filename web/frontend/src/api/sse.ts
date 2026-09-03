/** Line-delimited SSE reader over fetch, so the token can travel in a header.
 *
 * EventSource cannot set Authorization, and the alternative — a token in the query
 * string — would land in access logs. This trades ~50 lines for that.
 */

import { authHeaders, type RunEvent } from "./client";

export interface StreamHandle {
  close(): void;
}

interface StreamOptions {
  since: number;
  onEvent(event: RunEvent): void;
  onError?(error: unknown): void;
  onOpen?(): void;
}

const TERMINAL = new Set(["job.finished", "job.failed", "job.cancelled"]);

export function streamEvents(jobId: string, options: StreamOptions): StreamHandle {
  const controller = new AbortController();
  let closed = false;
  let since = options.since;
  let backoff = 1000;

  async function connect(): Promise<void> {
    while (!closed) {
      try {
        const response = await fetch(`/api/runs/${jobId}/events?since=${since}`, {
          headers: { ...authHeaders(), Accept: "text/event-stream" },
          signal: controller.signal,
        });
        if (!response.ok || !response.body) {
          throw new Error(`stream failed (${response.status})`);
        }
        options.onOpen?.();
        backoff = 1000;

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let done = false;

        while (!closed) {
          const chunk = await reader.read();
          if (chunk.done) break;
          buffer += decoder.decode(chunk.value, { stream: true });

          // Frames are blank-line separated; a partial tail stays in the buffer.
          let split: number;
          while ((split = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, split);
            buffer = buffer.slice(split + 2);
            for (const line of frame.split("\n")) {
              if (!line.startsWith("data:")) continue; // ": keepalive" and friends
              try {
                const event = JSON.parse(line.slice(5).trim()) as RunEvent;
                since = Math.max(since, event.seq);
                options.onEvent(event);
                if (TERMINAL.has(event.type)) done = true;
              } catch {
                // A torn frame is dropped; the cursor makes the next read authoritative.
              }
            }
          }
          if (done) break;
        }

        reader.cancel().catch(() => undefined);
        // The server closes the stream when a run ends, so a clean end is not an error.
        return;
      } catch (error) {
        if (closed || controller.signal.aborted) return;
        options.onError?.(error);
        await new Promise((resolve) => setTimeout(resolve, backoff));
        backoff = Math.min(backoff * 2, 15000);
      }
    }
  }

  void connect();

  return {
    close() {
      closed = true;
      controller.abort();
    },
  };
}
