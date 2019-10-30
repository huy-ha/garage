#!/usr/bin/env bash
status=0

echo "Checking commit range ${TRAVIS_COMMIT}...${TRAVIS_BRANCH}"
pre-commit run --source "${TRAVIS_COMMIT}" --origin "${TRAVIS_BRANCH}"
status="$((${status} | ${?}))"

# Check commit messages
echo "Checking commit messages"
while read commit; do
  commit_msg="$(mktemp)"
  git log --format=%B -n 1 "${commit}" > "${commit_msg}"
  pre-commit run --hook-stage commit-msg --commit-msg-file="${commit_msg}"
  pass=$?
  status="$((${status} | ${pass}))"

  # Print message if it fails
  if [[ "${pass}" -ne 0 ]]; then
    echo "Failing commit message:"
    cat "${commit_msg}"
  fi

done < <(git log --cherry-pick --left-only --pretty="%H" \
                 "${TRAVIS_COMMIT}...${TRAVIS_BRANCH}")

exit "${status}"
