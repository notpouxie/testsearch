CREATE OR REPLACE FUNCTION pg_dot_product(float[], float[])
RETURNS float AS 'MODULE_PATHNAME','testsearch'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_run_session(text)
RETURNS float[] AS 'MODULE_PATHNAME','testsearch'
LANGUAGE C STRICT;

