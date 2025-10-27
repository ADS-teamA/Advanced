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
    console.log('🌐 API: Making POST request to /chat/stream', { message, conversationId, model });

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

    console.log('📡 API: Response received', { ok: response.ok, status: response.status });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          console.log('📡 API: Stream done (reader finished)');
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        console.log('📦 API: Received chunk:', chunk.substring(0, 100) + (chunk.length > 100 ? '...' : ''));
        buffer += chunk;

        console.log('🔍 Buffer now contains', buffer.length, 'chars, ends with:', JSON.stringify(buffer.slice(-10)));

        // Process complete SSE messages
        // Note: Backend sends CRLF (\r\n\r\n) so we need to split on that
        const lines = buffer.split(/\r?\n\r?\n/);

        console.log('✂️ Split result:', lines.length, 'parts');

        // If split returned only 1 element, buffer doesn't contain separator yet
        // Keep waiting for more data
        if (lines.length === 1) {
          console.log('⏳ API: No complete SSE messages yet, waiting for more data...');
          continue;
        }

        // Last element is either incomplete or empty (if buffer ends with separator)
        buffer = lines.pop() || '';

        console.log(`📋 API: Processing ${lines.length} complete SSE messages, remaining buffer:`, buffer.length, 'chars');

        for (const line of lines) {
          if (!line.trim()) continue;

          console.log('📄 API: Processing line:', line);

          // Parse SSE format:
          // event: text_delta
          // data: {"content": "Hello"}
          const eventMatch = line.match(/^event:\s*(.+)$/m);
          const dataMatch = line.match(/^data:\s*(.+)$/m);

          if (eventMatch && dataMatch) {
            const event = eventMatch[1].trim();
            const data = JSON.parse(dataMatch[1]);

            console.log('✅ API: Parsed event:', event, 'data:', data);

            yield {
              event: event as SSEEvent['event'],
              data,
            };
          } else {
            console.warn('⚠️ API: Could not parse SSE line:', { line, eventMatch, dataMatch });
          }
        }
      }

      console.log('🏁 API: Stream loop finished');
    } finally {
      reader.releaseLock();
      console.log('🔓 API: Reader released');
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
