/**
 * API Service for backend communication
 *
 * Handles:
 * - SSE streaming for chat
 * - Model switching
 * - Health checks
 */

import type { Model, HealthStatus, SSEEvent } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export class ApiService {
  /**
   * Stream chat response using Server-Sent Events (SSE)
   */
  async *streamChat(
    message: string,
    conversationId?: string,
    model?: string
  ): AsyncGenerator<SSEEvent, void, unknown> {
    let response;
    try {
      response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          conversation_id: conversationId,
          model,
        }),
      });
    } catch (error) {
      console.error('❌ API: Fetch failed:', error);
      throw error;
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    // Timeout to prevent hanging on backend issues (5 minutes)
    const STREAM_TIMEOUT_MS = 5 * 60 * 1000;
    const timeoutId = setTimeout(() => {
      reader.cancel('Stream timeout after 5 minutes');
      console.error('⏰ API: Stream timeout - cancelling reader');
    }, STREAM_TIMEOUT_MS);

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Process complete SSE messages
        // Note: Backend sends CRLF (\r\n\r\n) so we need to split on that
        const lines = buffer.split(/\r?\n\r?\n/);

        // If split returned only 1 element, buffer doesn't contain separator yet
        // Keep waiting for more data
        if (lines.length === 1) {
          continue;
        }

        // Last element is either incomplete or empty (if buffer ends with separator)
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          // Parse SSE format:
          // event: text_delta
          // data: {"content": "Hello"}
          const eventMatch = line.match(/^event:\s*(.+)$/m);
          const dataMatch = line.match(/^data:\s*(.+)$/m);

          if (eventMatch && dataMatch) {
            const event = eventMatch[1].trim();

            // Try to parse JSON data
            try {
              const data = JSON.parse(dataMatch[1]);

              yield {
                event: event as SSEEvent['event'],
                data,
              };
            } catch (parseError) {
              // JSON parsing failed - this is a serious error, not just a warning
              console.error('❌ API: Failed to parse JSON in SSE data field:', {
                event,
                rawData: dataMatch[1],
                error: parseError
              });

              // Yield error event to surface the issue to UI
              yield {
                event: 'error',
                data: {
                  error: `Failed to parse server response (event: ${event})`,
                  type: 'JSONParseError',
                  rawData: dataMatch[1].substring(0, 100)
                }
              };
            }
          } else {
            // SSE format is invalid - this should NOT be just a warning
            console.error('❌ API: Invalid SSE format - missing event or data field:', {
              line,
              hasEvent: !!eventMatch,
              hasData: !!dataMatch
            });

            // Yield error event instead of silently dropping
            yield {
              event: 'error',
              data: {
                error: 'Server sent malformed response',
                type: 'SSEFormatError',
                details: line.substring(0, 100)
              }
            };
          }
        }
      }
    } finally {
      clearTimeout(timeoutId);
      reader.releaseLock();
    }
  }

  /**
   * Get list of available models
   */
  async getModels(): Promise<{ models: Model[]; defaultModel: string }> {
    const response = await fetch(`${API_BASE_URL}/models`);

    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Switch to a different model
   */
  async switchModel(model: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/model/switch?model=${encodeURIComponent(model)}`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to switch model: ${response.status}`);
    }
  }

  /**
   * Check backend health
   */
  async checkHealth(): Promise<HealthStatus> {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }

    return response.json();
  }
}

// Singleton instance
export const apiService = new ApiService();
