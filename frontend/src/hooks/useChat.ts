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
  const [selectedModel, setSelectedModel] = useState<string>(() =>
    // Load from localStorage, or use Gemini 2.5 Flash Lite as default
    storageService.getSelectedModel() || 'gemini-2.5-flash-latest-exp-1206'
  );

  // Refs for managing streaming state
  const currentMessageRef = useRef<Message | null>(null);
  const currentToolCallsRef = useRef<Map<string, ToolCall>>(new Map());

  // Ref to always access latest conversations (avoid stale closure)
  const conversationsRef = useRef<Conversation[]>(conversations);

  // Update ref when conversations change
  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);

  // Initialize and verify model
  useEffect(() => {
    const savedModel = storageService.getSelectedModel();

    // If no saved model, save the default
    if (!savedModel) {
      storageService.setSelectedModel('gemini-2.5-flash-latest-exp-1206');
    }

    // Verify model is available on backend
    apiService.getModels().then((data) => {
      const currentModel = savedModel || 'gemini-2.5-flash-latest-exp-1206';

      // If current model is not available, fallback to backend default
      if (!data.models.some((m: any) => m.id === currentModel)) {
        const defaultModel = data.defaultModel || 'gemini-2.5-flash-latest-exp-1206';
        setSelectedModel(defaultModel);
        storageService.setSelectedModel(defaultModel);
      }
    }).catch(console.error);
  }, []);

  /**
   * Clean invalid/incomplete messages from conversation
   * Prevents corrupted data in localStorage
   */
  const cleanMessages = useCallback((messages: Message[]): Message[] => {
    return messages.filter(msg =>
      msg &&
      msg.role &&
      msg.id &&
      msg.timestamp &&
      msg.content !== undefined
    );
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
   * @param content - The message content to send
   * @param addUserMessage - Whether to add a new user message (false for regenerate/edit)
   */
  const sendMessage = useCallback(
    async (content: string, addUserMessage: boolean = true) => {
      if (isStreaming || !content.trim()) {
        console.log('❌ Early return:', { isStreaming, contentEmpty: !content.trim() });
        return;
      }

      // Get current conversation using currentConversationId
      let conversation: Conversation;

      if (currentConversationId) {
        // Find conversation from state using functional read
        const found = conversations.find((c) => c.id === currentConversationId);
        if (found) {
          conversation = found;
          console.log('🔵 sendMessage using existing conversation:', conversation.id);
        } else {
          console.log('📝 Creating new conversation (ID not found)');
          conversation = createConversation();
        }
      } else {
        console.log('📝 Creating new conversation (no current ID)');
        conversation = createConversation();
      }

      let updatedConversation: Conversation;

      if (addUserMessage) {
        // Create and add user message
        const userMessage: Message = {
          id: `msg_${Date.now()}_user`,
          role: 'user',
          content: content.trim(),
          timestamp: new Date(),
        };

        // Add user message to conversation
        updatedConversation = {
          ...conversation,
          messages: [...conversation.messages, userMessage],
          updatedAt: new Date(),
          title: conversation.messages.length === 0 ? content.slice(0, 50) : conversation.title,
        };
      } else {
        // Regenerate/edit mode - use conversation as-is
        updatedConversation = {
          ...conversation,
          updatedAt: new Date(),
        };
      }

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
              id: event.data.call_id || `tool_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
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
          } else if (event.event === 'tool_calls_summary') {
            // Tool calls summary from backend
            if (currentMessageRef.current && event.data.tool_calls) {
              // Convert backend tool calls to frontend format
              currentMessageRef.current.toolCalls = event.data.tool_calls.map((tc: any) => ({
                id: tc.id,
                name: tc.name,
                input: tc.input,
                result: tc.result,
                executionTimeMs: tc.executionTimeMs,
                success: tc.success,
                status: tc.success === false ? 'failed' as const : 'completed' as const,
                explicitParams: tc.explicitParams,
              }));

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

        // Ensure final message is in state before saving
        // Capture the ref value before async operations
        const finalMessage = currentMessageRef.current;
        if (finalMessage) {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== updatedConversation.id) return c;

              const messages = [...c.messages];
              const lastMsg = messages[messages.length - 1];

              // Update or add final assistant message
              if (lastMsg?.role === 'assistant') {
                messages[messages.length - 1] = { ...finalMessage };
              } else {
                messages.push({ ...finalMessage });
              }

              // Clean messages - remove any incomplete/invalid messages
              const cleanedMessages = messages.filter(msg =>
                msg &&
                msg.role &&
                msg.id &&
                msg.timestamp &&
                msg.content !== undefined
              );

              const finalConv = { ...c, messages: cleanedMessages, updatedAt: new Date() };

              // Save to localStorage with final content
              console.log('💾 Saving final conversation to localStorage:', finalConv.messages.length, 'messages');
              console.log('💬 Final assistant message content length:', finalMessage.content.length);
              storageService.saveConversation(finalConv);

              return finalConv;
            })
          );
        }
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
    [isStreaming, createConversation, selectedModel, currentConversationId, conversations]
  );

  /**
   * Switch model
   */
  const switchModel = useCallback(async (model: string) => {
    try {
      await apiService.switchModel(model);
      setSelectedModel(model);
      storageService.setSelectedModel(model);
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
      if (isStreaming || !newContent.trim()) return;

      // Use functional update to get current conversation state
      let shouldSend = false;

      setConversations((prev) => {
        // Find the current conversation
        const currentConv = prev.find((c) => c.id === currentConversationId);
        if (!currentConv) return prev;

        // Find the message index
        const messageIndex = currentConv.messages.findIndex((m) => m.id === messageId);
        if (messageIndex === -1 || currentConv.messages[messageIndex].role !== 'user') {
          return prev;
        }

        shouldSend = true;

        // Remove all messages after this one (including assistant response)
        const updatedMessages = currentConv.messages.slice(0, messageIndex);

        // Clean messages before saving
        const cleanedMessages = cleanMessages(updatedMessages);

        // Update conversation with truncated messages
        const updatedConversation: Conversation = {
          ...currentConv,
          messages: cleanedMessages,
          updatedAt: new Date(),
        };

        // Save to storage immediately
        storageService.saveConversation(updatedConversation);

        // Return updated conversations array
        return prev.map((c) => (c.id === currentConversationId ? updatedConversation : c));
      });

      // If validation passed, send the edited message
      if (shouldSend) {
        // Small delay to ensure React has processed the state update
        await new Promise(resolve => setTimeout(resolve, 10));
        await sendMessage(newContent);
      }
    },
    [isStreaming, sendMessage, currentConversationId]
  );

  /**
   * Regenerate the last assistant response
   */
  const regenerateMessage = useCallback(
    async (messageId: string) => {
      console.log('🔥 regenerateMessage called with messageId:', messageId);
      console.log('🔥 isStreaming:', isStreaming);
      console.log('🔥 currentConversationId:', currentConversationId);

      if (isStreaming) {
        console.log('🔥 Aborting: isStreaming is true');
        return;
      }

      // SYNCHRONOUSLY read current conversation from ref (always fresh)
      console.log('🔥 Reading conversations from ref, count:', conversationsRef.current.length);
      const currentConv = conversationsRef.current.find((c) => c.id === currentConversationId);

      if (!currentConv) {
        console.log('🔥 ERROR: Current conversation not found!');
        return;
      }

      console.log('🔥 Found conversation, messages count:', currentConv.messages.length);

      // Find the message index
      const messageIndex = currentConv.messages.findIndex((m) => m.id === messageId);
      console.log('🔥 Message index:', messageIndex);

      if (messageIndex === -1) {
        console.log('🔥 ERROR: Message not found!');
        return;
      }

      const message = currentConv.messages[messageIndex];
      console.log('🔥 Message role:', message.role);

      if (message.role !== 'assistant') {
        console.log('🔥 ERROR: Message is not from assistant!');
        return;
      }

      // Find the user message before this assistant message
      const userMessageIndex = messageIndex - 1;
      console.log('🔥 User message index:', userMessageIndex);

      if (userMessageIndex < 0) {
        console.log('🔥 ERROR: No user message before assistant message!');
        return;
      }

      const userMessage = currentConv.messages[userMessageIndex];
      console.log('🔥 User message role:', userMessage.role);

      if (userMessage.role !== 'user') {
        console.log('🔥 ERROR: Previous message is not from user!');
        return;
      }

      const userMessageContent = userMessage.content;
      const conversationId = currentConv.id;
      console.log('🔥 Success! Will regenerate with user message:', userMessageContent.substring(0, 50));

      // Remove all messages after (and including) the assistant message we want to regenerate
      const updatedMessages = currentConv.messages.slice(0, messageIndex);

      // Clean messages before saving
      const cleanedMessages = cleanMessages(updatedMessages);

      // Update conversation
      const updatedConversation: Conversation = {
        ...currentConv,
        messages: cleanedMessages,
        updatedAt: new Date(),
      };

      // Update state
      setConversations((prev) =>
        prev.map((c) => (c.id === conversationId ? updatedConversation : c))
      );

      // Save to storage
      storageService.saveConversation(updatedConversation);

      console.log('🔥 Calling sendMessage with:', userMessageContent.substring(0, 50));
      // Small delay to ensure React has processed the state update
      await new Promise(resolve => setTimeout(resolve, 10));
      // Pass false to prevent adding a new user message (we're regenerating from existing)
      await sendMessage(userMessageContent, false);
      console.log('🔥 sendMessage completed');
    },
    [isStreaming, sendMessage, currentConversationId]
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
