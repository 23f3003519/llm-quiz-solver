# LLM Quiz Solver API

A FastAPI application that automatically solves data science quizzes using Selenium for JavaScript rendering and LLM analysis.

## Features
- JavaScript-rendered quiz page scraping
- Base64 content decoding
- PDF processing and analysis
- Multi-URL quiz chain handling
- Prompt security testing

## API Endpoints
- `POST /solve` - Main quiz solving endpoint
- `POST /test-prompt` - Prompt security testing
- `GET /debug-test` - System diagnostics

## Deployment
Deployed on Vercel: `https://your-app.vercel.app`