# Senior Project Chat

A full-stack AI chat application built with Next.js, Neon PostgreSQL, and the Hugging Face Inference API. Users can register, log in, and have persistent conversations with an AI model served via a Hugging Face endpoint.

## Architecture

```
Browser (React)  <-->  Next.js API Routes  <-->  Neon PostgreSQL
                                |
                                v
                     Hugging Face Inference API
```

The app is a single Next.js project. The frontend and backend live together — Next.js API routes (`src/app/api/`) serve as the backend, so there is no separate server to deploy.

### Frontend (`src/app/`)

- **`/login` and `/register`** — Simple username/password auth forms. On success, a session token is stored in `localStorage` and sent as a `Bearer` token on all subsequent API requests.
- **`/chat`** — The main interface. A sidebar lists past conversations, and the main area displays the active conversation with streaming AI responses. Users can create new conversations, switch between them, and send messages.

### Backend (`src/app/api/`)

| Route | Method | Description |
|---|---|---|
| `/api/auth/register` | POST | Creates a new user (bcrypt-hashed password) and returns a session token |
| `/api/auth/login` | POST | Validates credentials and returns a session token |
| `/api/conversations` | GET | Lists all conversations for the authenticated user |
| `/api/conversations` | POST | Creates a new conversation |
| `/api/conversations/[id]/messages` | GET | Returns all messages in a conversation |
| `/api/chat` | POST | Saves the user message, calls the HF Inference API with the full conversation history, streams the response back via Server-Sent Events, and saves the assistant reply to the database |

### Database (Neon PostgreSQL)

Four tables:

- **`users`** — id, username, password_hash, created_at
- **`sessions`** — id, user_id (FK), token, created_at
- **`conversations`** — id, user_id (FK), title, created_at, updated_at
- **`messages`** — id, conversation_id (FK), role (user/assistant), content, created_at

### AI Inference

The `/api/chat` route proxies requests to a Hugging Face Inference endpoint using the OpenAI-compatible chat completions format. Responses are streamed back to the client in real time using Server-Sent Events (SSE). The model and endpoint URL are configurable via environment variables, making it straightforward to swap in a custom model.

## Setup

### Prerequisites

- Node.js 18+
- A Neon PostgreSQL database
- A Hugging Face API token

### Install and run

```bash
# Install dependencies
npm install

# Copy the example env file and fill in your values
cp .env.example .env.local

# Create the database tables
npm run setup-db

# Start the dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to use the app.

### Environment variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | Neon PostgreSQL connection string |
| `HF_API_TOKEN` | Hugging Face API token |
| `HF_BASE_URL` | Inference API base URL (default: `https://router.huggingface.co/v1`) |
| `HF_MODEL` | Model ID to use for chat completions |
| `SESSION_SECRET` | Secret for session management |

### Swapping in a custom model

To use your own model deployed on a Hugging Face Inference Endpoint, update `.env.local`:

```
HF_BASE_URL=https://your-endpoint.endpoints.huggingface.cloud/v1
HF_MODEL=your-model-name
```

The endpoint must expose an OpenAI-compatible `/chat/completions` route, which Hugging Face Inference Endpoints do by default (via TGI or vLLM).

## Deployment

Deploy to Vercel:

1. Push this repo to GitHub
2. Import the repo at [vercel.com](https://vercel.com)
3. Add the environment variables from `.env.local` in the Vercel project settings
4. Deploy

The database migration (`npm run setup-db`) only needs to be run once.
