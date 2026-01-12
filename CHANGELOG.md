# CHANGELOG

<!-- version list -->

## v1.1.0 (2026-01-12)

### Bug Fixes

- **ci**: Add commit check and increase verbosity for debugging
  ([`b91e2f4`](https://github.com/darvid/ultrasync/commit/b91e2f41f812c7d47df15c5a67f8da7dfd666e0f))

- **ci**: Add commit_author to match SSH signing identity
  ([`c7c9b4b`](https://github.com/darvid/ultrasync/commit/c7c9b4ba926082705815f42699b27da3fd1c69c7))

- **ci**: Add git_committer_name and email for SSH signing
  ([`e49fed8`](https://github.com/darvid/ultrasync/commit/e49fed8df734685ee7a2aad227e839624ab562c4))

- **ci**: Align release author with signing key
  ([`23cdb9a`](https://github.com/darvid/ultrasync/commit/23cdb9afeb39c3c075524918841c9e14fc08a13c))

- **ci**: Enable release tagging
  ([`cf424a8`](https://github.com/darvid/ultrasync/commit/cf424a8ecc54d134b263e2290911b20580ec7c9c))

- **ci**: Remove gitsign (runs in Docker, doesn't work)
  ([`9ba3266`](https://github.com/darvid/ultrasync/commit/9ba3266f1177685b0a957dd8efdc5af2caa7a1d0))

- **ci**: Restore PR-based release workflow with GPG signing
  ([`2691d07`](https://github.com/darvid/ultrasync/commit/2691d07245b25d815c9e3db3e2ac8288c8e84d99))

- **ci**: Run semantic-release on main branch before creating release branch
  ([`3253cfa`](https://github.com/darvid/ultrasync/commit/3253cfa51b33beba927a6e56295d1152e51ce95e))

- **ci**: Skip PR creation when no commits between branches
  ([`4e6a1ff`](https://github.com/darvid/ultrasync/commit/4e6a1ffa8ae49780ee5ee099b992efe11e8bf3dd))

- **ci**: Update semantic-release workflow for v10
  ([`43317d8`](https://github.com/darvid/ultrasync/commit/43317d8eacb5dbcf454fd69a4f942cd249f3ff78))

- **ci**: Use darvid noreply for release commits
  ([`68b8a35`](https://github.com/darvid/ultrasync/commit/68b8a3525aef964c7acc4102da4cb0aaaad2b030))

- **ci**: Use GitHub App token for signed commits
  ([`57ea129`](https://github.com/darvid/ultrasync/commit/57ea129518f493c780488468fa00817a497011d9))

- **ci**: Use GitHub App token for signed commits
  ([`a5f387c`](https://github.com/darvid/ultrasync/commit/a5f387cba1e74938b9185d05f08ed5b658d36bac))

- **ci**: Use gitsign for keyless commit signing
  ([`b623f8e`](https://github.com/darvid/ultrasync/commit/b623f8e5185eb3ee14228827cffb8fc049fcf544))

- **ci**: Use integer for semantic-release verbosity
  ([`f82a837`](https://github.com/darvid/ultrasync/commit/f82a8376f873c984614709162c042b9b426f76f3))

- **ci**: Use SSH signing for verified commits
  ([`a024a59`](https://github.com/darvid/ultrasync/commit/a024a5916c1661a1bb89e5d3888a03b4b6a2c487))

- **mcp**: Default root to cwd when not specified
  ([`fff440f`](https://github.com/darvid/ultrasync/commit/fff440f796ee9e396ea664f6d9fff02800eca8b5))

- **memory**: Preserve transcript timestamps instead of using current time
  ([`bf4ccf0`](https://github.com/darvid/ultrasync/commit/bf4ccf046f1b4bef0f298b64695c06b28f4ca075))

- **paths**: Default to global XDG directory instead of per-project
  ([`39b81e7`](https://github.com/darvid/ultrasync/commit/39b81e7166f6ce2f6c6086ba36d7dac5e32ed56e))

- **sync**: Receive user_id from server welcome event
  ([`8b1da65`](https://github.com/darvid/ultrasync/commit/8b1da65e77584cd79d5e5c19c4668138472afa9d))

### Features

- **paths**: Make data directory configurable with XDG defaults
  ([`67f2aa4`](https://github.com/darvid/ultrasync/commit/67f2aa4f291c18e521a7ab3cd8c312e3a618e04d))


## v1.0.1 (2026-01-01)

### Bug Fixes

- **ci**: Remove release trigger to prevent duplicate publish attempts
  ([`c47e050`](https://github.com/darvid/ultrasync/commit/c47e0503f31e52f7f3c09482c87ad0cbafe1cda0))

- **ci**: Skip PR creation if release PR already exists
  ([`1bf4b6b`](https://github.com/darvid/ultrasync/commit/1bf4b6b4c0a19417662e3e9a8d064a9a2d6bd01e))

- **ci**: Use semantic-release changelog for GitHub releases
  ([`807d6b1`](https://github.com/darvid/ultrasync/commit/807d6b120e97459487de52ca255426783a3e6d16))


## v1.0.0 (2025-12-31)

- Initial Release
