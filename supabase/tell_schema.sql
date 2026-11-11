-- Bootstrap a fresh Tell Supabase project. Do not apply over an existing Tell database.
begin;

create schema if not exists private;
grant usage on schema private to authenticated;

create table public.tell_teams (
  id uuid primary key default gen_random_uuid(),
  name text not null check (char_length(name) between 1 and 100),
  join_code text not null unique check (join_code ~ '^[A-Z]{6}$')
);

create table public.tell_members (
  team_id uuid not null references public.tell_teams(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('owner', 'member')),
  member_name text,
  primary key (team_id, user_id)
);
create index tell_members_by_user on public.tell_members (user_id, team_id);

create table public.tell_items (
  team_id uuid not null references public.tell_teams(id) on delete cascade,
  kind text not null check (kind in ('player', 'match', 'attack')),
  id text not null,
  payload jsonb not null,
  deleted boolean not null default false,
  updated_at timestamptz not null default clock_timestamp(),
  primary key (team_id, kind, id)
);
create index tell_items_sync on public.tell_items (team_id, updated_at, kind, id);

alter table public.tell_teams enable row level security;
alter table public.tell_members enable row level security;
alter table public.tell_items enable row level security;
revoke all on public.tell_teams, public.tell_members, public.tell_items from anon, authenticated;
grant select on public.tell_teams to authenticated;
grant select, delete on public.tell_members to authenticated;
grant select, insert, update on public.tell_items to authenticated;

create function private.tell_has_team(team uuid)
returns boolean
language sql stable security definer set search_path = ''
as $$
  select exists (
    select 1 from public.tell_members where team_id = team and user_id = (select auth.uid())
  );
$$;
revoke all on function private.tell_has_team(uuid) from public;
grant execute on function private.tell_has_team(uuid) to authenticated;

create policy tell_teams_read on public.tell_teams
  for select to authenticated using (private.tell_has_team(id));
create policy tell_members_read on public.tell_members
  for select to authenticated using (user_id = (select auth.uid()));
create policy tell_members_leave on public.tell_members
  for delete to authenticated using (user_id = (select auth.uid()));
create policy tell_items_team on public.tell_items
  for all to authenticated using (private.tell_has_team(team_id))
  with check (private.tell_has_team(team_id));

create function private.tell_stamp_item()
returns trigger language plpgsql set search_path = '' as $$
begin
  new.updated_at := clock_timestamp();
  return new;
end;
$$;
revoke all on function private.tell_stamp_item() from public, anon, authenticated;
create trigger tell_stamp_item before insert or update on public.tell_items
  for each row execute function private.tell_stamp_item();

create function private.tell_create_team(team_name text, member_name text default null)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare
  requester uuid := auth.uid();
  team public.tell_teams;
  random_bytes bytea;
  new_code text;
begin
  if requester is null then raise exception 'Sign in to create a team'; end if;
  if team_name is null or char_length(trim(team_name)) not between 1 and 100 then
    raise exception 'Enter a team name between 1 and 100 characters';
  end if;

  loop
    random_bytes := decode(replace(gen_random_uuid()::text, '-', ''), 'hex');
    select string_agg(chr(65 + get_byte(random_bytes, byte_index) % 26), '' order by byte_index)
      into new_code from generate_series(0, 5) as bytes(byte_index);
    insert into public.tell_teams (name, join_code) values (trim(team_name), new_code)
      on conflict (join_code) do nothing returning * into team;
    exit when found;
  end loop;

  insert into public.tell_members (team_id, user_id, role, member_name)
    values (team.id, requester, 'owner', nullif(trim(member_name), ''));
  return to_jsonb(team);
end;
$$;

create function private.tell_join_team(code text, member_name text default null)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare
  requester uuid := auth.uid();
  team public.tell_teams;
begin
  if requester is null then raise exception 'Sign in to join a team'; end if;
  select * into team from public.tell_teams where join_code = upper(trim(code));
  if not found then raise exception 'No team has that code'; end if;

  insert into public.tell_members (team_id, user_id, role, member_name)
    values (team.id, requester, 'member', nullif(trim(member_name), ''))
    on conflict (team_id, user_id) do nothing;
  return to_jsonb(team);
end;
$$;

revoke all on function private.tell_create_team(text, text), private.tell_join_team(text, text) from public, anon;
grant execute on function private.tell_create_team(text, text), private.tell_join_team(text, text) to authenticated;

create function public.tell_create_team(team_name text, member_name text default null)
returns jsonb language sql security invoker set search_path = '' as $$
  select private.tell_create_team(team_name, member_name);
$$;

create function public.tell_join_team(code text, member_name text default null)
returns jsonb language sql security invoker set search_path = '' as $$
  select private.tell_join_team(code, member_name);
$$;

revoke all on function public.tell_create_team(text, text), public.tell_join_team(text, text) from public, anon;
grant execute on function public.tell_create_team(text, text), public.tell_join_team(text, text) to authenticated;

commit;
