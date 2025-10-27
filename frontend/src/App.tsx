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

import { useState } from 'react';
import { Header } from './components/header/Header';
import { Sidebar } from './components/sidebar/Sidebar';
import { ResponsiveSidebar } from './components/layout/ResponsiveSidebar';
import { ChatContainer } from './components/chat/ChatContainer';
import { useChat } from './hooks/useChat';
import { useTheme } from './hooks/useTheme';
import { cn } from './design-system/utils/cn';
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

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

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
