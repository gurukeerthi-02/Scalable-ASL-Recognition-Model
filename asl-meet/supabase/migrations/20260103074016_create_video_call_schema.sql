-- Video Call Application Schema
-- Updated to support flexible 6-character Room IDs

-- Create rooms table
CREATE TABLE IF NOT EXISTS rooms (
  id text PRIMARY KEY, -- Changed from UUID to support short codes
  name text DEFAULT '',
  created_at timestamptz DEFAULT now(),
  max_participants integer DEFAULT 4,
  created_by text DEFAULT '',
  is_active boolean DEFAULT true
);

-- Create participants table
CREATE TABLE IF NOT EXISTS participants (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  room_id text NOT NULL REFERENCES rooms(id) ON DELETE CASCADE, -- Text reference
  peer_id text NOT NULL,
  display_name text DEFAULT 'Anonymous',
  joined_at timestamptz DEFAULT now(),
  left_at timestamptz,
  is_asl_enabled boolean DEFAULT false
);

-- Create call_logs table
CREATE TABLE IF NOT EXISTS call_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  room_id text NOT NULL REFERENCES rooms(id) ON DELETE CASCADE, -- Text reference
  participant_id uuid REFERENCES participants(id) ON DELETE SET NULL,
  event_type text NOT NULL,
  event_data jsonb,
  created_at timestamptz DEFAULT now()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_participants_room_id ON participants(room_id);
CREATE INDEX IF NOT EXISTS idx_participants_peer_id ON participants(peer_id);
CREATE INDEX IF NOT EXISTS idx_call_logs_room_id ON call_logs(room_id);
CREATE INDEX IF NOT EXISTS idx_call_logs_created_at ON call_logs(created_at DESC);

-- Enable Row Level Security
ALTER TABLE rooms ENABLE ROW LEVEL SECURITY;
ALTER TABLE participants ENABLE ROW LEVEL SECURITY;
ALTER TABLE call_logs ENABLE ROW LEVEL SECURITY;

-- Simple public policies for research/demo
CREATE POLICY "Anyone can view rooms" ON rooms FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "Anyone can create rooms" ON rooms FOR INSERT TO anon, authenticated WITH CHECK (true);
CREATE POLICY "Anyone can update rooms" ON rooms FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Anyone can view participants" ON participants FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "Anyone can create participants" ON participants FOR INSERT TO anon, authenticated WITH CHECK (true);
CREATE POLICY "Anyone can update participants" ON participants FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Anyone can delete participants" ON participants FOR DELETE TO anon, authenticated USING (true);

CREATE POLICY "Anyone can view call logs" ON call_logs FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "Anyone can create call logs" ON call_logs FOR INSERT TO anon, authenticated WITH CHECK (true);