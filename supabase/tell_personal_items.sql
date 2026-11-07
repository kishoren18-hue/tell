-- Apply before deploying the app version that enables private account sync.
begin;

create table if not exists public.tell_personal_items (
  user_id uuid not null references auth.users (id) on delete cascade,
  kind text not null check (kind in ('player', 'match', 'attack')),
  id text not null,
  payload jsonb not null,
  deleted boolean not null default false,
  updated_at timestamptz not null default clock_timestamp(),
  primary key (user_id, kind, id)
);

create index if not exists tell_personal_items_sync_idx
  on public.tell_personal_items (user_id, updated_at, kind, id);

alter table public.tell_personal_items enable row level security;
revoke all on public.tell_personal_items from anon, authenticated;
grant select, insert, update on public.tell_personal_items to authenticated;

drop policy if exists tell_personal_items_owner on public.tell_personal_items;
create policy tell_personal_items_owner on public.tell_personal_items
  for all to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);

create or replace function public.tell_personal_stamp()
returns trigger language plpgsql set search_path = '' as $$
begin
  new.updated_at := clock_timestamp();
  return new;
end;
$$;

drop trigger if exists tell_personal_stamp on public.tell_personal_items;
create trigger tell_personal_stamp before insert or update on public.tell_personal_items
  for each row execute function public.tell_personal_stamp();

commit;
