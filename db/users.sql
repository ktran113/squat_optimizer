CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    name TEXT,
    created TIMESTAMP DEFAULT now(),
    updated TIMESTAMP DEFUALT now()
);
CREATE INDEX idx_users_email ON users(email);
