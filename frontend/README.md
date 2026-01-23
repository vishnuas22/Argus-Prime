# Argus Prime Frontend

Production-ready Next.js 14 frontend for the Argus Prime Multi-Modal Deepfake Detection Platform.

## 🚀 Quick Start

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## 📦 Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript 5.x
- **Styling**: Tailwind CSS 3.x
- **State Management**: Zustand 4.x
- **Data Fetching**: TanStack Query 5.x
- **Animations**: Framer Motion 11.x
- **Charts**: Recharts 2.x
- **Icons**: Lucide React
- **Testing**: Vitest + Playwright

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── globals.css        # Global styles
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx           # Landing page
│   ├── components/
│   │   └── ui/               # Base UI components (atoms)
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       └── progress.tsx
│   ├── lib/                   # Utilities and configurations
│   │   ├── api.ts            # API client
│   │   ├── config.ts         # App configuration
│   │   └── utils.ts          # Helper functions
│   ├── hooks/                 # Custom React hooks
│   ├── stores/                # Global state stores
│   ├── types/                 # TypeScript types
│   └── contexts/              # React contexts
├── public/                    # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── next.config.js
└── README.md
```

## 🎨 Design System

### Colors
- **Primary**: Indigo/Purple gradient (primary-500 to primary-600)
- **Secondary**: Neutral gray scale
- **Semantic**: Success (emerald), Warning (amber), Danger (red)

### Typography
- **Sans**: Inter (Google Fonts)
- **Mono**: JetBrains Mono

### Components
All base UI components follow the design specifications in `Argus_Premium_Frontend_GLM.md`:
- Button (5 variants: primary, secondary, outline, ghost, danger)
- Card (4 variants: default, elevated, outlined, glass)
- Progress (linear and circular variants)

## 🔌 API Integration

The frontend integrates with the Argus Prime backend via:

- **REST API**: Analysis submission, status polling, results retrieval
- **WebSocket**: Real-time progress streaming

### Key Endpoints
- `POST /api/v1/analyze` - Submit file for analysis
- `GET /api/v1/analyze/{id}/status` - Get analysis progress
- `GET /api/v1/analyze/{id}/result` - Get analysis results
- `WS /api/v1/analyze/{id}/stream` - Real-time progress

See `src/lib/api.ts` for the complete API client implementation.

## 🧪 Testing

```bash
# Run unit tests
npm run test

# Run tests with UI
npm run test:ui

# Run E2E tests
npx playwright install
npx playwright test
```

## 📝 Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm run typecheck` | Run TypeScript type checking |
| `npm run test` | Run unit tests |
| `npm run test:ci` | Run tests in CI mode |

## 🚢 Deployment

### Vercel (Recommended)
```bash
npm i -g vercel
vercel
```

### Docker
```bash
docker build -t argus-prime-frontend .
docker run -p 3000:3000 argus-prime-frontend
```

## 📄 Documentation

- **Frontend Guide**: [Argus_Premium_Frontend_GLM.md](../../Argus_Premium_Frontend_GLM.md)
- **Backend API**: [PRIME_ARGUS_DOCUMENT.md](../../PRIME_ARGUS_DOCUMENT.md)
- **Frontend Constitution**: [AGENTS_FRONTEND.md](../../AGENTS_FRONTEND.md)

## 🤝 Contributing

1. Follow the coding conventions in `AGENTS_FRONTEND.md`
2. Ensure all tests pass before submitting
3. Maintain component documentation
4. Add unit tests for new components

## 📜 License

MIT License - See LICENSE file for details.
