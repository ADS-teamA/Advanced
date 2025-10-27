# SUJBOT2 Frontend

React + TypeScript frontend for SUJBOT2 web interface with ChatGPT/Claude-like UI.

## Features

- 🎨 Modern chat interface with real-time streaming
- ⚡ Server-Sent Events (SSE) for instant responses
- 🌓 Dark/Light mode with system preference detection
- 💬 Conversation history (persisted in LocalStorage)
- 🛠️ Tool call visualization with expandable details
- 💰 Token usage and cost tracking per message
- 📝 Full markdown rendering with GitHub Flavored Markdown
- 🎯 Syntax highlighting for code blocks
- 🤖 Model switching (Claude, GPT models)

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server (Vite with HMR)
npm run dev
```

Open **http://localhost:5173** in your browser.

## Project Structure

```
src/
├── components/          # React components
│   ├── chat/           # Chat UI components
│   │   ├── ChatMessage.tsx       # Single message display
│   │   ├── ChatInput.tsx         # Message input area
│   │   ├── ChatContainer.tsx     # Main chat area
│   │   └── ToolCallDisplay.tsx   # Tool execution display
│   ├── sidebar/        # Conversation sidebar
│   │   └── Sidebar.tsx           # Conversation list
│   └── header/         # Top navigation
│       └── Header.tsx            # Model selector, theme toggle
├── hooks/              # Custom React hooks
│   ├── useChat.ts      # Chat state & SSE streaming
│   └── useTheme.ts     # Dark/light mode
├── services/           # API clients
│   └── api.ts          # Backend API & SSE client
├── types/              # TypeScript definitions
│   └── index.ts        # All interfaces
├── lib/                # Utilities
│   └── storage.ts      # LocalStorage management
├── App.tsx             # Main app component
├── main.tsx            # React entry point
└── index.css           # Tailwind CSS + custom styles
```

## Technology Stack

- **React 18.3** - UI framework with hooks
- **TypeScript 5.6** - Type safety
- **Vite 7** - Build tool with instant HMR
- **Tailwind CSS 3.4** - Utility-first styling
- **Lucide React** - Beautiful icon library
- **react-markdown 9** - Markdown rendering
- **rehype-highlight** - Code syntax highlighting
- **remark-gfm** - GitHub Flavored Markdown support

## Development

```bash
# Development server with hot reload
npm run dev

# Type checking
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

## Architecture

### State Management

Uses **React hooks** without Redux/Zustand:
- `useChat` - Manages conversation state and SSE streaming
- `useTheme` - Manages dark/light mode
- LocalStorage for persistence

### SSE Streaming Flow

```
User Input → useChat.sendMessage()
    ↓
apiService.streamChat() (SSE async generator)
    ↓
Yields events: text_delta, tool_call, tool_result, cost_update
    ↓
useChat updates state → UI re-renders in real-time
```

### Data Persistence

All conversations stored in browser LocalStorage:
- `sujbot_conversations` - All conversation data
- `sujbot_current_conversation` - Active conversation ID
- `sujbot_theme` - Theme preference

## API Integration

Backend API: **http://localhost:8000**

### SSE Event Format

```typescript
interface SSEEvent {
  event: 'text_delta' | 'tool_call' | 'tool_result' | 'cost_update' | 'done' | 'error';
  data: any;
}
```

See `src/services/api.ts` for full API client implementation.

## Customization

### Colors

Edit `tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      // Add custom colors here
    }
  }
}
```

### Fonts

Edit `src/index.css`:

```css
:root {
  font-family: 'Your Font', system-ui, sans-serif;
}
```

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

Requires: ES2020, SSE support, LocalStorage API

## Performance

- **Initial bundle**: ~500KB (after gzip)
- **Vite HMR**: <100ms update time
- **SSE latency**: <50ms for events
- **LocalStorage limit**: 5-10MB (browser dependent)

## Troubleshooting

### Cannot connect to backend

1. Check backend is running: `http://localhost:8000/health`
2. Verify CORS is enabled in backend
3. Check browser console for network errors

### LocalStorage quota exceeded

Clear storage in browser console:
```javascript
localStorage.clear()
```

Or delete old conversations via sidebar.

### Styling not working

Rebuild Tailwind:
```bash
npm run build
```

## Contributing

1. Follow existing component patterns
2. Use TypeScript strict mode
3. Keep components under 200 lines
4. Extract complex logic to hooks
5. Add PropTypes documentation

## License

Same as main SUJBOT2 project.
