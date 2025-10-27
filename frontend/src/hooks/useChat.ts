/**
 * useChat hook - Manages chat state and SSE streaming
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { apiService } from '../services/api';
import { storageService } from '../lib/storage';
import type { Message, Conversation, ToolCall } from '../types';

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>(() =>
    storageService.getConversations()
  );
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(() =>
    storageService.getCurrentConversationId()
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('claude-haiku-4-5-20251001');

  // Refs for managing streaming state
  const currentMessageRef = useRef<Message | null>(null);
  const currentToolCallsRef = useRef<Map<string, ToolCall>>(new Map());

  // Load default model from backend on mount (or use Haiku 4.5 as default)
  useEffect(() => {
    apiService.getModels().then((data) => {
      // Use Haiku 4.5 as default, or backend's default if different
      setSelectedModel(data.defaultModel || 'claude-haiku-4-5-20251001');
    }).catch(console.error);
  }, []);

  /**
   * Get current conversation
   */
  const currentConversation = conversations.find((c) => c.id === currentConversationId);

  /**
   * Create a new conversation
   */
  const createConversation = useCallback(() => {
    const newConversation: Conversation = {
      id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title: 'New Conversation',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    setConversations((prev) => [...prev, newConversation]);
    setCurrentConversationId(newConversation.id);
    storageService.saveConversation(newConversation);
    storageService.setCurrentConversationId(newConversation.id);

    return newConversation;
  }, []);

  /**
   * Select a conversation
   */
  const selectConversation = useCallback((id: string) => {
    setCurrentConversationId(id);
    storageService.setCurrentConversationId(id);
  }, []);

  /**
   * Delete a conversation
   */
  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    storageService.deleteConversation(id);

    if (currentConversationId === id) {
      setCurrentConversationId(null);
    }
  }, [currentConversationId]);

  /**
   * Send a message and stream the response
   */
  const sendMessage = useCallback(
    async (content: string) => {
      console.log('🔵 sendMessage called with:', { content, isStreaming, hasConversation: !!currentConversation });

      if (isStreaming || !content.trim()) {
        console.log('❌ Early return:', { isStreaming, contentEmpty: !content.trim() });
        return;
      }

      // Ensure we have a conversation
      let conversation = currentConversation;
      if (!conversation) {
        console.log('📝 Creating new conversation');
        conversation = createConversation();
      }

      // Create user message
      const userMessage: Message = {
        id: `msg_${Date.now()}_user`,
        role: 'user',
        content: content.trim(),
        timestamp: new Date(),
      };

      // Add user message to conversation
      const updatedConversation: Conversation = {
        ...conversation,
        messages: [...conversation.messages, userMessage],
        updatedAt: new Date(),
        title: conversation.messages.length === 0 ? content.slice(0, 50) : conversation.title,
      };

      setConversations((prev) =>
        prev.map((c) => (c.id === updatedConversation.id ? updatedConversation : c))
      );
      storageService.saveConversation(updatedConversation);

      // Initialize assistant message
      currentMessageRef.current = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        toolCalls: [],
      };
      currentToolCallsRef.current = new Map();

      setIsStreaming(true);

      try {
        console.log('🚀 Starting API stream:', { conversationId: conversation.id, model: selectedModel });

        // Stream response from backend
        for await (const event of apiService.streamChat(
          content,
          conversation.id,
          selectedModel
        )) {
          console.log('📨 Received event:', event.event);
          if (event.event === 'text_delta') {
            // Append text delta
            if (currentMessageRef.current) {
              currentMessageRef.current.content += event.data.content;

              console.log('💬 Text delta:', event.data.content);
              console.log('📝 Current message content now:', currentMessageRef.current.content.substring(0, 50));
              console.log('🕐 Current message timestamp:', currentMessageRef.current.timestamp);

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  console.log('🔍 Last message role:', lastMsg?.role);

                  if (lastMsg?.role === 'assistant') {
                    // Update existing assistant message
                    console.log('✏️ Updating existing assistant message');
                    messages[messages.length - 1] = { ...currentMessageRef.current! };
                  } else {
                    // Add new assistant message
                    console.log('➕ Adding new assistant message');
                    messages.push({ ...currentMessageRef.current! });
                  }

                  console.log('📨 Total messages now:', messages.length);

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'tool_call') {
            // Tool execution started
            const toolCall: ToolCall = {
              id: event.data.call_id,
              name: event.data.tool_name,
              input: event.data.tool_input,
              status: 'running',
            };

            currentToolCallsRef.current.set(toolCall.id, toolCall);

            if (currentMessageRef.current) {
              currentMessageRef.current.toolCalls = Array.from(
                currentToolCallsRef.current.values()
              );
            }

            // Update UI
            setConversations((prev) =>
              prev.map((c) => {
                if (c.id !== updatedConversation.id) return c;

                const messages = [...c.messages];
                const lastMsg = messages[messages.length - 1];

                if (lastMsg?.role === 'assistant') {
                  messages[messages.length - 1] = { ...currentMessageRef.current! };
                } else {
                  messages.push({ ...currentMessageRef.current! });
                }

                return { ...c, messages };
              })
            );
          } else if (event.event === 'tool_result') {
            // Tool execution completed
            const existingToolCall = currentToolCallsRef.current.get(event.data.call_id);

            if (existingToolCall) {
              existingToolCall.result = event.data.result;
              existingToolCall.executionTimeMs = event.data.execution_time_ms;
              existingToolCall.success = event.data.success;
              existingToolCall.status = event.data.success ? 'completed' : 'failed';

              if (currentMessageRef.current) {
                currentMessageRef.current.toolCalls = Array.from(
                  currentToolCallsRef.current.values()
                );
              }

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  messages[messages.length - 1] = { ...currentMessageRef.current! };

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'cost_update') {
            // Cost tracking update
            if (currentMessageRef.current) {
              currentMessageRef.current.cost = {
                totalCost: event.data.total_cost,
                inputTokens: event.data.input_tokens,
                outputTokens: event.data.output_tokens,
                cachedTokens: event.data.cached_tokens,
                summary: event.data.summary,
              };

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  messages[messages.length - 1] = { ...currentMessageRef.current! };

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'done') {
            // Stream completed
            break;
          } else if (event.event === 'error') {
            // Error occurred
            console.error('Stream error:', event.data);

            if (currentMessageRef.current) {
              currentMessageRef.current.content += `\n\n[Error: ${event.data.error}]`;
            }
          }
        }

        // Save final conversation state
        // Use setConversations to get the latest state
        setConversations((prev) => {
          const finalConversation = prev.find((c) => c.id === updatedConversation.id);
          if (finalConversation) {
            console.log('💾 Saving final conversation to localStorage:', finalConversation.messages.length, 'messages');
            storageService.saveConversation(finalConversation);
          }
          return prev; // Don't modify state, just save
        });
      } catch (error) {
        console.error('❌ Error during streaming:', error);
        console.error('Error stack:', (error as Error).stack);

        // Add error message
        if (currentMessageRef.current) {
          currentMessageRef.current.content += `\n\n[Error: ${(error as Error).message}]`;
        }
      } finally {
        console.log('✅ Stream finished, cleaning up');
        setIsStreaming(false);
        currentMessageRef.current = null;
        currentToolCallsRef.current = new Map();
      }
    },
    [isStreaming, currentConversation, createConversation, selectedModel, conversations]
  );

  /**
   * Switch model
   */
  const switchModel = useCallback(async (model: string) => {
    try {
      await apiService.switchModel(model);
      setSelectedModel(model);
    } catch (error) {
      console.error('Failed to switch model:', error);
      throw error;
    }
  }, []);

  /**
   * Edit a user message and resend
   */
  const editMessage = useCallback(
    async (messageId: string, newContent: string) => {
      if (!currentConversation || isStreaming || !newContent.trim()) return;

      // Find the message index
      const messageIndex = currentConversation.messages.findIndex((m) => m.id === messageId);
      if (messageIndex === -1 || currentConversation.messages[messageIndex].role !== 'user') {
        return;
      }

      // Remove all messages after this one (including assistant response)
      const updatedMessages = currentConversation.messages.slice(0, messageIndex);

      // Update conversation with truncated messages
      const updatedConversation: Conversation = {
        ...currentConversation,
        messages: updatedMessages,
        updatedAt: new Date(),
      };

      setConversations((prev) =>
        prev.map((c) => (c.id === updatedConversation.id ? updatedConversation : c))
      );
      storageService.saveConversation(updatedConversation);

      // Send the edited message
      await sendMessage(newContent);
    },
    [currentConversation, isStreaming, sendMessage]
  );

  /**
   * Regenerate the last assistant response
   */
  const regenerateMessage = useCallback(
    async (messageId: string) => {
      if (!currentConversation || isStreaming) return;

      // Find the message index
      const messageIndex = currentConversation.messages.findIndex((m) => m.id === messageId);
      if (messageIndex === -1 || currentConversation.messages[messageIndex].role !== 'assistant') {
        return;
      }

      // Find the user message before this assistant message
      const userMessageIndex = messageIndex - 1;
      if (userMessageIndex < 0 || currentConversation.messages[userMessageIndex].role !== 'user') {
        return;
      }

      const userMessage = currentConversation.messages[userMessageIndex];

      // Remove the assistant message
      const updatedMessages = currentConversation.messages.slice(0, messageIndex);

      // Update conversation
      const updatedConversation: Conversation = {
        ...currentConversation,
        messages: updatedMessages,
        updatedAt: new Date(),
      };

      setConversations((prev) =>
        prev.map((c) => (c.id === updatedConversation.id ? updatedConversation : c))
      );
      storageService.saveConversation(updatedConversation);

      // Resend the user message to regenerate response
      await sendMessage(userMessage.content);
    },
    [currentConversation, isStreaming, sendMessage]
  );

  return {
    conversations,
    currentConversation,
    isStreaming,
    selectedModel,
    createConversation,
    selectConversation,
    deleteConversation,
    sendMessage,
    switchModel,
    editMessage,
    regenerateMessage,
  };
}
