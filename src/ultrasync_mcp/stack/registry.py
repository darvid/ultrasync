from __future__ import annotations

from dataclasses import dataclass, field

from ultrasync_mcp.ir import ComponentKind


@dataclass
class ComponentMeta:
    kind: ComponentKind
    features: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


REGISTRY: dict[str, ComponentMeta] = {
    # === JavaScript/TypeScript Frameworks ===
    "next": ComponentMeta(
        ComponentKind.FRAMEWORK,
        [
            "app-router",
            "pages-router",
            "api-routes",
            "middleware",
            "server-actions",
        ],
    ),
    "react": ComponentMeta(ComponentKind.FRAMEWORK),
    "react-dom": ComponentMeta(ComponentKind.FRAMEWORK),
    "vue": ComponentMeta(ComponentKind.FRAMEWORK),
    "nuxt": ComponentMeta(ComponentKind.FRAMEWORK),
    "svelte": ComponentMeta(ComponentKind.FRAMEWORK),
    "@sveltejs/kit": ComponentMeta(ComponentKind.FRAMEWORK),
    "solid-js": ComponentMeta(ComponentKind.FRAMEWORK),
    "angular": ComponentMeta(ComponentKind.FRAMEWORK),
    "@angular/core": ComponentMeta(ComponentKind.FRAMEWORK),
    "express": ComponentMeta(ComponentKind.FRAMEWORK),
    "hono": ComponentMeta(ComponentKind.FRAMEWORK),
    "elysia": ComponentMeta(ComponentKind.FRAMEWORK),
    "koa": ComponentMeta(ComponentKind.FRAMEWORK),
    "fastify": ComponentMeta(ComponentKind.FRAMEWORK),
    "nest": ComponentMeta(ComponentKind.FRAMEWORK),
    "@nestjs/core": ComponentMeta(ComponentKind.FRAMEWORK),
    "remix": ComponentMeta(ComponentKind.FRAMEWORK),
    "@remix-run/node": ComponentMeta(ComponentKind.FRAMEWORK),
    "astro": ComponentMeta(ComponentKind.FRAMEWORK),
    # === Python Frameworks ===
    "fastapi": ComponentMeta(ComponentKind.FRAMEWORK),
    "flask": ComponentMeta(ComponentKind.FRAMEWORK),
    "django": ComponentMeta(ComponentKind.FRAMEWORK),
    "starlette": ComponentMeta(ComponentKind.FRAMEWORK),
    "litestar": ComponentMeta(ComponentKind.FRAMEWORK),
    "sanic": ComponentMeta(ComponentKind.FRAMEWORK),
    "tornado": ComponentMeta(ComponentKind.FRAMEWORK),
    "aiohttp": ComponentMeta(ComponentKind.FRAMEWORK),
    # === ORM / Database Libraries ===
    "prisma": ComponentMeta(
        ComponentKind.LIBRARY,
        ["postgres", "mysql", "sqlite", "mongodb", "cockroachdb"],
    ),
    "@prisma/client": ComponentMeta(ComponentKind.LIBRARY),
    "drizzle-orm": ComponentMeta(
        ComponentKind.LIBRARY,
        ["postgres", "mysql", "sqlite"],
    ),
    "typeorm": ComponentMeta(ComponentKind.LIBRARY),
    "sequelize": ComponentMeta(ComponentKind.LIBRARY),
    "mongoose": ComponentMeta(ComponentKind.LIBRARY),
    "kysely": ComponentMeta(ComponentKind.LIBRARY),
    "knex": ComponentMeta(ComponentKind.LIBRARY),
    "sqlalchemy": ComponentMeta(
        ComponentKind.LIBRARY,
        ["postgres", "mysql", "sqlite", "async"],
    ),
    "tortoise-orm": ComponentMeta(ComponentKind.LIBRARY),
    "peewee": ComponentMeta(ComponentKind.LIBRARY),
    "sqlmodel": ComponentMeta(ComponentKind.LIBRARY),
    "alembic": ComponentMeta(ComponentKind.LIBRARY),
    # === Validation Libraries ===
    "zod": ComponentMeta(ComponentKind.LIBRARY),
    "yup": ComponentMeta(ComponentKind.LIBRARY),
    "joi": ComponentMeta(ComponentKind.LIBRARY),
    "valibot": ComponentMeta(ComponentKind.LIBRARY),
    "ajv": ComponentMeta(ComponentKind.LIBRARY),
    "pydantic": ComponentMeta(ComponentKind.LIBRARY),
    "marshmallow": ComponentMeta(ComponentKind.LIBRARY),
    "cerberus": ComponentMeta(ComponentKind.LIBRARY),
    # === State Management ===
    "zustand": ComponentMeta(ComponentKind.LIBRARY),
    "jotai": ComponentMeta(ComponentKind.LIBRARY),
    "recoil": ComponentMeta(ComponentKind.LIBRARY),
    "redux": ComponentMeta(ComponentKind.LIBRARY),
    "@reduxjs/toolkit": ComponentMeta(ComponentKind.LIBRARY),
    "mobx": ComponentMeta(ComponentKind.LIBRARY),
    "xstate": ComponentMeta(ComponentKind.LIBRARY),
    # === HTTP / API Clients ===
    "axios": ComponentMeta(ComponentKind.LIBRARY),
    "ky": ComponentMeta(ComponentKind.LIBRARY),
    "got": ComponentMeta(ComponentKind.LIBRARY),
    "node-fetch": ComponentMeta(ComponentKind.LIBRARY),
    "httpx": ComponentMeta(ComponentKind.LIBRARY),
    "requests": ComponentMeta(ComponentKind.LIBRARY),
    # === Auth Libraries ===
    "next-auth": ComponentMeta(ComponentKind.LIBRARY),
    "@auth/core": ComponentMeta(ComponentKind.LIBRARY),
    "lucia": ComponentMeta(ComponentKind.LIBRARY),
    "passport": ComponentMeta(ComponentKind.LIBRARY),
    "jsonwebtoken": ComponentMeta(ComponentKind.LIBRARY),
    "jose": ComponentMeta(ComponentKind.LIBRARY),
    "bcrypt": ComponentMeta(ComponentKind.LIBRARY),
    "argon2": ComponentMeta(ComponentKind.LIBRARY),
    "python-jose": ComponentMeta(ComponentKind.LIBRARY),
    "pyjwt": ComponentMeta(ComponentKind.LIBRARY),
    "passlib": ComponentMeta(ComponentKind.LIBRARY),
    # === UI Component Libraries ===
    "@radix-ui/react-dialog": ComponentMeta(ComponentKind.LIBRARY),
    "@radix-ui/react-dropdown-menu": ComponentMeta(ComponentKind.LIBRARY),
    "@headlessui/react": ComponentMeta(ComponentKind.LIBRARY),
    "@mantine/core": ComponentMeta(ComponentKind.LIBRARY),
    "@chakra-ui/react": ComponentMeta(ComponentKind.LIBRARY),
    "@mui/material": ComponentMeta(ComponentKind.LIBRARY),
    "antd": ComponentMeta(ComponentKind.LIBRARY),
    # === CSS / Styling ===
    "tailwindcss": ComponentMeta(ComponentKind.LIBRARY),
    "postcss": ComponentMeta(ComponentKind.LIBRARY),
    "autoprefixer": ComponentMeta(ComponentKind.LIBRARY),
    "sass": ComponentMeta(ComponentKind.LIBRARY),
    "styled-components": ComponentMeta(ComponentKind.LIBRARY),
    "@emotion/react": ComponentMeta(ComponentKind.LIBRARY),
    # === External Service Adapters ===
    "stripe": ComponentMeta(
        ComponentKind.ADAPTER,
        ["payments", "subscriptions", "webhooks", "connect", "billing"],
    ),
    "@stripe/stripe-js": ComponentMeta(ComponentKind.ADAPTER),
    "resend": ComponentMeta(ComponentKind.ADAPTER, ["email"]),
    "@sendgrid/mail": ComponentMeta(ComponentKind.ADAPTER, ["email"]),
    "nodemailer": ComponentMeta(ComponentKind.ADAPTER, ["email"]),
    "twilio": ComponentMeta(ComponentKind.ADAPTER, ["sms", "voice"]),
    "@clerk/nextjs": ComponentMeta(ComponentKind.ADAPTER, ["auth"]),
    "@clerk/clerk-sdk-node": ComponentMeta(ComponentKind.ADAPTER, ["auth"]),
    "@supabase/supabase-js": ComponentMeta(
        ComponentKind.ADAPTER,
        ["auth", "database", "storage", "realtime"],
    ),
    "firebase": ComponentMeta(ComponentKind.ADAPTER),
    "firebase-admin": ComponentMeta(ComponentKind.ADAPTER),
    "@aws-sdk/client-s3": ComponentMeta(ComponentKind.ADAPTER, ["storage"]),
    "@aws-sdk/client-ses": ComponentMeta(ComponentKind.ADAPTER, ["email"]),
    "@aws-sdk/client-sqs": ComponentMeta(ComponentKind.ADAPTER, ["queue"]),
    "@aws-sdk/client-dynamodb": ComponentMeta(
        ComponentKind.ADAPTER, ["database"]
    ),
    "@azure/storage-blob": ComponentMeta(ComponentKind.ADAPTER, ["storage"]),
    "@google-cloud/storage": ComponentMeta(ComponentKind.ADAPTER, ["storage"]),
    "openai": ComponentMeta(
        ComponentKind.ADAPTER, ["llm", "embeddings", "chat"]
    ),
    "anthropic": ComponentMeta(ComponentKind.ADAPTER, ["llm", "chat"]),
    "@anthropic-ai/sdk": ComponentMeta(ComponentKind.ADAPTER, ["llm", "chat"]),
    "langchain": ComponentMeta(
        ComponentKind.ADAPTER, ["llm", "agents", "chains"]
    ),
    "@langchain/core": ComponentMeta(ComponentKind.ADAPTER),
    "llamaindex": ComponentMeta(ComponentKind.ADAPTER, ["llm", "rag"]),
    "pinecone": ComponentMeta(ComponentKind.ADAPTER, ["vector-db"]),
    "@pinecone-database/pinecone": ComponentMeta(
        ComponentKind.ADAPTER, ["vector-db"]
    ),
    "chromadb": ComponentMeta(ComponentKind.ADAPTER, ["vector-db"]),
    "qdrant-client": ComponentMeta(ComponentKind.ADAPTER, ["vector-db"]),
    "weaviate-client": ComponentMeta(ComponentKind.ADAPTER, ["vector-db"]),
    # === Database Drivers ===
    "pg": ComponentMeta(ComponentKind.ADAPTER, ["postgres"]),
    "postgres": ComponentMeta(ComponentKind.ADAPTER, ["postgres"]),
    "@neondatabase/serverless": ComponentMeta(
        ComponentKind.ADAPTER, ["postgres"]
    ),
    "mysql2": ComponentMeta(ComponentKind.ADAPTER, ["mysql"]),
    "better-sqlite3": ComponentMeta(ComponentKind.ADAPTER, ["sqlite"]),
    "mongodb": ComponentMeta(ComponentKind.ADAPTER, ["mongodb"]),
    "redis": ComponentMeta(ComponentKind.ADAPTER, ["redis"]),
    "ioredis": ComponentMeta(ComponentKind.ADAPTER, ["redis"]),
    "@upstash/redis": ComponentMeta(ComponentKind.ADAPTER, ["redis"]),
    "psycopg2": ComponentMeta(ComponentKind.ADAPTER, ["postgres"]),
    "psycopg": ComponentMeta(ComponentKind.ADAPTER, ["postgres"]),
    "asyncpg": ComponentMeta(ComponentKind.ADAPTER, ["postgres"]),
    "pymongo": ComponentMeta(ComponentKind.ADAPTER, ["mongodb"]),
    "motor": ComponentMeta(ComponentKind.ADAPTER, ["mongodb"]),
    "aioredis": ComponentMeta(ComponentKind.ADAPTER, ["redis"]),
    # === Queue / Message Brokers ===
    "bullmq": ComponentMeta(ComponentKind.ADAPTER, ["queue"]),
    "bee-queue": ComponentMeta(ComponentKind.ADAPTER, ["queue"]),
    "amqplib": ComponentMeta(ComponentKind.ADAPTER, ["rabbitmq"]),
    "kafkajs": ComponentMeta(ComponentKind.ADAPTER, ["kafka"]),
    "celery": ComponentMeta(ComponentKind.ADAPTER, ["queue"]),
    "rq": ComponentMeta(ComponentKind.ADAPTER, ["queue"]),
    "aio-pika": ComponentMeta(ComponentKind.ADAPTER, ["rabbitmq"]),
    # === Dev Utilities ===
    "typescript": ComponentMeta(ComponentKind.UTILITY),
    "eslint": ComponentMeta(ComponentKind.UTILITY),
    "prettier": ComponentMeta(ComponentKind.UTILITY),
    "biome": ComponentMeta(ComponentKind.UTILITY),
    "@biomejs/biome": ComponentMeta(ComponentKind.UTILITY),
    "husky": ComponentMeta(ComponentKind.UTILITY),
    "lint-staged": ComponentMeta(ComponentKind.UTILITY),
    "commitlint": ComponentMeta(ComponentKind.UTILITY),
    # === Testing ===
    "vitest": ComponentMeta(ComponentKind.UTILITY),
    "jest": ComponentMeta(ComponentKind.UTILITY),
    "@testing-library/react": ComponentMeta(ComponentKind.UTILITY),
    "playwright": ComponentMeta(ComponentKind.UTILITY),
    "@playwright/test": ComponentMeta(ComponentKind.UTILITY),
    "cypress": ComponentMeta(ComponentKind.UTILITY),
    "mocha": ComponentMeta(ComponentKind.UTILITY),
    "chai": ComponentMeta(ComponentKind.UTILITY),
    "pytest": ComponentMeta(ComponentKind.UTILITY),
    "pytest-asyncio": ComponentMeta(ComponentKind.UTILITY),
    "pytest-cov": ComponentMeta(ComponentKind.UTILITY),
    "hypothesis": ComponentMeta(ComponentKind.UTILITY),
    "unittest": ComponentMeta(ComponentKind.UTILITY),
    "ruff": ComponentMeta(ComponentKind.UTILITY),
    "black": ComponentMeta(ComponentKind.UTILITY),
    "isort": ComponentMeta(ComponentKind.UTILITY),
    "mypy": ComponentMeta(ComponentKind.UTILITY),
    "pyright": ComponentMeta(ComponentKind.UTILITY),
    # === Build Tools ===
    "vite": ComponentMeta(ComponentKind.UTILITY),
    "esbuild": ComponentMeta(ComponentKind.UTILITY),
    "rollup": ComponentMeta(ComponentKind.UTILITY),
    "webpack": ComponentMeta(ComponentKind.UTILITY),
    "turbo": ComponentMeta(ComponentKind.UTILITY),
    "tsup": ComponentMeta(ComponentKind.UTILITY),
    "unbuild": ComponentMeta(ComponentKind.UTILITY),
}


def infer_kind(name: str) -> ComponentKind:
    if name.startswith("@types/"):
        return ComponentKind.UTILITY
    if any(
        x in name.lower()
        for x in ["eslint", "prettier", "lint", "test", "mock"]
    ):
        return ComponentKind.UTILITY
    if any(x in name.lower() for x in ["sdk", "client", "driver", "adapter"]):
        return ComponentKind.ADAPTER
    if name.startswith("@aws-sdk/") or name.startswith("@azure/"):
        return ComponentKind.ADAPTER
    if name.startswith("@google-cloud/"):
        return ComponentKind.ADAPTER
    return ComponentKind.LIBRARY
