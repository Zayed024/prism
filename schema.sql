-- Prism: Multi-Agent Productivity Assistant
-- Database schema for AlloyDB/PostgreSQL

-- Enable AlloyDB AI extensions
CREATE EXTENSION IF NOT EXISTS google_ml_integration CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;

-- Register Gemini model for in-database AI
-- CALL google_ml.create_model(
--     model_id => 'gemini-embedding',
--     model_request_url => 'https://aiplatform.googleapis.com/v1/projects/responsive-amp-438114-j0/locations/us-central1/publishers/google/models/text-embedding-005:predict',
--     model_qualified_name => 'text-embedding-005',
--     model_provider => 'google',
--     model_type => 'text_embedding',
--     model_auth_type => 'alloydb_service_agent_iam'
-- );

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    description TEXT DEFAULT '',
    status VARCHAR(20) DEFAULT 'todo' CHECK (status IN ('todo', 'in_progress', 'done')),
    priority VARCHAR(10) DEFAULT 'medium' CHECK (priority IN ('high', 'medium', 'low')),
    due_date TIMESTAMP,
    tags TEXT[] DEFAULT '{}',
    created_by VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Notes table
CREATE TABLE IF NOT EXISTS notes (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT DEFAULT '',
    tags TEXT[] DEFAULT '{}',
    linked_task_id INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    created_by VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Prism session logs
CREATE TABLE IF NOT EXISTS prism_sessions (
    id SERIAL PRIMARY KEY,
    user_request TEXT NOT NULL,
    red_response JSONB,
    blue_response JSONB,
    green_response JSONB,
    merged_result JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Agent performance tracking
CREATE TABLE IF NOT EXISTS agent_performance (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES prism_sessions(id),
    agent_name VARCHAR(20) NOT NULL,
    was_selected BOOLEAN DEFAULT FALSE,
    tools_used TEXT[] DEFAULT '{}',
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Embedding columns for AlloyDB AI semantic search
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS embedding vector(768);
ALTER TABLE notes ADD COLUMN IF NOT EXISTS embedding vector(768);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_notes_linked_task ON notes(linked_task_id);
CREATE INDEX IF NOT EXISTS idx_perf_session ON agent_performance(session_id);

-- Vector indexes for semantic search (HNSW for fast approximate nearest neighbor)
-- CREATE INDEX IF NOT EXISTS idx_tasks_embedding ON tasks USING hnsw (embedding vector_cosine_ops);
-- CREATE INDEX IF NOT EXISTS idx_notes_embedding ON notes USING hnsw (embedding vector_cosine_ops);

-- Function: Generate embedding using AlloyDB AI (google_ml.predict_row)
-- This calls Vertex AI's text-embedding model directly from the database
CREATE OR REPLACE FUNCTION generate_embedding(input_text TEXT)
RETURNS vector AS $$
DECLARE
    result JSON;
    embedding_array FLOAT[];
BEGIN
    SELECT google_ml.predict_row(
        model_id => 'gemini-embedding',
        request_body => json_build_object(
            'instances', json_build_array(
                json_build_object('content', input_text)
            )
        )::json
    ) INTO result;

    SELECT ARRAY(
        SELECT json_array_elements_text(
            result->'predictions'->0->'embeddings'->'values'
        )::FLOAT
    ) INTO embedding_array;

    RETURN embedding_array::vector;
END;
$$ LANGUAGE plpgsql;

-- Function: Semantic search across tasks
CREATE OR REPLACE FUNCTION semantic_search_tasks(query_text TEXT, match_limit INT DEFAULT 5)
RETURNS TABLE(id INT, title VARCHAR, description TEXT, status VARCHAR, priority VARCHAR, similarity FLOAT) AS $$
DECLARE
    query_embedding vector;
BEGIN
    query_embedding := generate_embedding(query_text);
    RETURN QUERY
        SELECT t.id, t.title, t.description, t.status, t.priority,
               1 - (t.embedding <=> query_embedding)::FLOAT AS similarity
        FROM tasks t
        WHERE t.embedding IS NOT NULL
        ORDER BY t.embedding <=> query_embedding
        LIMIT match_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Semantic search across notes
CREATE OR REPLACE FUNCTION semantic_search_notes(query_text TEXT, match_limit INT DEFAULT 5)
RETURNS TABLE(id INT, title VARCHAR, content TEXT, similarity FLOAT) AS $$
DECLARE
    query_embedding vector;
BEGIN
    query_embedding := generate_embedding(query_text);
    RETURN QUERY
        SELECT n.id, n.title, n.content,
               1 - (n.embedding <=> query_embedding)::FLOAT AS similarity
        FROM notes n
        WHERE n.embedding IS NOT NULL
        ORDER BY n.embedding <=> query_embedding
        LIMIT match_limit;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-generate embeddings on insert/update
CREATE OR REPLACE FUNCTION auto_embed_task() RETURNS TRIGGER AS $$
BEGIN
    NEW.embedding := generate_embedding(NEW.title || ' ' || COALESCE(NEW.description, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION auto_embed_note() RETURNS TRIGGER AS $$
BEGIN
    NEW.embedding := generate_embedding(NEW.title || ' ' || COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- DROP TRIGGER IF EXISTS trg_task_embed ON tasks;
-- CREATE TRIGGER trg_task_embed BEFORE INSERT OR UPDATE OF title, description ON tasks
--     FOR EACH ROW EXECUTE FUNCTION auto_embed_task();

-- DROP TRIGGER IF EXISTS trg_note_embed ON notes;
-- CREATE TRIGGER trg_note_embed BEFORE INSERT OR UPDATE OF title, content ON notes
--     FOR EACH ROW EXECUTE FUNCTION auto_embed_note();

-- Seed data: Demo tasks
INSERT INTO tasks (title, description, status, priority, due_date, tags, created_by) VALUES
('Finish Q2 quarterly report', 'Compile data from all departments and create executive summary', 'in_progress', 'high', '2026-04-10 17:00:00', ARRAY['work', 'report', 'Q2'], 'user'),
('Review API documentation', 'Go through the updated API docs and flag inconsistencies', 'todo', 'medium', '2026-04-07 12:00:00', ARRAY['work', 'docs', 'api'], 'user'),
('Prepare client presentation', 'Create slide deck for Friday client meeting', 'todo', 'high', '2026-04-04 09:00:00', ARRAY['work', 'client', 'presentation'], 'user'),
('Grocery shopping', 'Buy vegetables, milk, eggs, and bread', 'todo', 'low', '2026-04-05 18:00:00', ARRAY['personal', 'errands'], 'user'),
('Code review: auth module', 'Review PR #247 for the new authentication flow', 'todo', 'high', '2026-04-04 15:00:00', ARRAY['work', 'code-review', 'auth'], 'user'),
('Schedule dentist appointment', 'Call Dr. Patel office for a cleaning', 'todo', 'low', NULL, ARRAY['personal', 'health'], 'user'),
('Write blog post on MCP', 'Draft a technical blog about Model Context Protocol integration patterns', 'in_progress', 'medium', '2026-04-12 12:00:00', ARRAY['work', 'writing', 'tech'], 'user'),
('Update team onboarding guide', 'Add new sections about CI/CD pipeline and testing', 'todo', 'medium', '2026-04-15 17:00:00', ARRAY['work', 'docs', 'onboarding'], 'user'),
('Plan weekend trip', 'Research destinations and book accommodation for Apr 18-20', 'todo', 'low', '2026-04-11 20:00:00', ARRAY['personal', 'travel'], 'user'),
('Fix deployment pipeline bug', 'Cloud Build fails intermittently on the test stage', 'in_progress', 'high', '2026-04-05 12:00:00', ARRAY['work', 'devops', 'bug'], 'user');

-- Seed data: Demo notes
INSERT INTO notes (title, content, tags, linked_task_id, created_by) VALUES
('Q2 Report Data Sources', E'## Data Sources\n- Finance: revenue_q2.xlsx (shared drive)\n- Engineering: velocity metrics from Jira\n- Marketing: campaign performance from Analytics\n- Sales: pipeline report from Salesforce\n\n## Key Deadlines\n- Draft due: Apr 8\n- Review: Apr 9\n- Final submission: Apr 10', ARRAY['report', 'Q2', 'data'], 1, 'user'),
('Client Meeting Agenda', E'## Friday Client Meeting\n- Project status update (15 min)\n- Demo new features (20 min)\n- Roadmap discussion (15 min)\n- Q&A (10 min)\n\n## Prep needed\n- Update demo environment\n- Prepare backup slides for technical questions', ARRAY['client', 'meeting', 'agenda'], 3, 'user'),
('Auth Module Notes', E'## Current Issues with PR #247\n- Token refresh logic needs edge case handling\n- Missing rate limiting on login endpoint\n- Session invalidation not tested\n\n## Suggested Changes\n- Add retry with exponential backoff\n- Implement sliding window rate limiter\n- Add integration tests for session lifecycle', ARRAY['code-review', 'auth', 'security'], 5, 'user'),
('MCP Blog Outline', E'## Title: Building Multi-Agent Systems with MCP\n\n1. What is Model Context Protocol?\n2. Why MCP matters for AI applications\n3. Building your first MCP server (code walkthrough)\n4. Connecting multiple MCP servers to an orchestrator\n5. Real-world patterns: tool routing, error handling\n6. Performance considerations\n7. Conclusion + resources', ARRAY['blog', 'MCP', 'draft'], 7, 'user'),
('Weekly Standup Notes - Mar 30', E'## What I did\n- Completed API endpoint refactoring\n- Fixed 3 critical bugs in notification service\n- Started Q2 report data collection\n\n## What I plan to do\n- Finish Q2 report draft\n- Review auth module PR\n- Client presentation prep\n\n## Blockers\n- Waiting on finance data for Q2 report\n- CI/CD pipeline intermittent failures', ARRAY['standup', 'weekly', 'status'], NULL, 'user');
