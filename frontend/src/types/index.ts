/**
 * TypeScript types for frontend application
 */

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];
  cost?: CostInfo;
}

export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, any>;
  result?: any;
  executionTimeMs?: number;
  success?: boolean;
  status: 'running' | 'completed' | 'failed';
}

export interface CostInfo {
  totalCost: number;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  summary: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface Model {
  id: string;
  name: string;
  provider: 'anthropic' | 'openai';
  description: string;
}

export interface SSEEvent {
  event: 'text_delta' | 'tool_call' | 'tool_result' | 'cost_update' | 'done' | 'error';
  data: any;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'error';
  message: string;
  details: Record<string, any>;
}
