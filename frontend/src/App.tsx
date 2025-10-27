/**
 * Main App Component
 *
 * Wires together:
 * - Header (with model selector, theme toggle, and sidebar toggle)
 * - ResponsiveSidebar (conversation history with collapsible behavior)
 * - ChatContainer (messages and input)
 *
 * Uses custom hooks:
 * - useChat: Manages conversation state and SSE streaming
 * - useTheme: Manages dark/light mode
 */

import { useState, useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Header } from './components/header/Header';
import { Sidebar } from './components/sidebar/Sidebar';
import { ResponsiveSidebar } from './components/layout/ResponsiveSidebar';
import { ChatContainer } from './components/chat/ChatContainer';
import { useChat } from './hooks/useChat';
import { useTheme } from './hooks/useTheme';
import { cn } from './design-system/utils/cn';
import { apiService } from './services/api';
import './index.css';

function App() {
  // Custom hooks
  const {
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
  } = useChat();

  const { theme, toggleTheme } = useTheme();

  // Sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Health status state
  const [degradedComponents, setDegradedComponents] = useState<Array<{component: string; error: string}>>([]);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  // Check health status on mount
  useEffect(() => {
    apiService.checkHealth()
      .then((health) => {
        if (health.status === 'degraded' && health.degraded_components) {
          setDegradedComponents(health.degraded_components);
        }
      })
      .catch((error) => {
        console.error('Health check failed:', error);
      });
  }, []);

  return (
    <div className={cn(
      'h-screen flex flex-col',
      'bg-white dark:bg-accent-950',
      'text-accent-900 dark:text-accent-100'
    )}>
      {/* Header */}
      <Header
        theme={theme}
        onToggleTheme={toggleTheme}
        selectedModel={selectedModel}
        onModelChange={switchModel}
        onToggleSidebar={toggleSidebar}
        sidebarOpen={sidebarOpen}
      />

      {/* Degraded Mode Warning Banner */}
      {degradedComponents.length > 0 && (
        <div className={cn(
          'px-4 py-3 border-b',
          'bg-yellow-50 dark:bg-yellow-900/20',
          'border-yellow-200 dark:border-yellow-800',
          'text-yellow-800 dark:text-yellow-200'
        )}>
          <div className="flex items-start gap-3 max-w-7xl mx-auto">
            <AlertTriangle size={20} className="flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <div className="font-semibold text-sm mb-1">
                Running in Degraded Mode
              </div>
              <div className="text-xs opacity-90">
                Some features are unavailable: {degradedComponents.map(d => d.component).join(', ')}.
                {' '}Search quality may be reduced without reranking. Knowledge graph features are disabled.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Responsive Sidebar */}
        <ResponsiveSidebar isOpen={sidebarOpen} onToggle={toggleSidebar}>
          <Sidebar
            conversations={conversations}
            currentConversationId={currentConversation?.id || null}
            onSelectConversation={selectConversation}
            onNewConversation={createConversation}
            onDeleteConversation={deleteConversation}
          />
        </ResponsiveSidebar>

        {/* Chat area */}
        <ChatContainer
          conversation={currentConversation}
          isStreaming={isStreaming}
          onSendMessage={sendMessage}
          onEditMessage={editMessage}
          onRegenerateMessage={regenerateMessage}
        />
      </div>
    </div>
  );
}

export default App;
