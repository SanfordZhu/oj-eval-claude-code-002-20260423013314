#include "include/int2048.h"

namespace sjtu {

class BigIntCore {
public:
  static const int base = 10000; // 1e4
  static const int base_digits = 4;
};

static void trim(std::vector<int> &a) {
  while (!a.empty() && a.back() == 0) a.pop_back();
}

static int cmp_abs(const std::vector<int> &a, const std::vector<int> &b) {
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (int i = (int)a.size() - 1; i >= 0; --i) {
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  }
  return 0;
}

static std::vector<int> add_abs(const std::vector<int> &a, const std::vector<int> &b) {
  const int base = BigIntCore::base;
  std::vector<int> res;
  res.reserve(std::max(a.size(), b.size()) + 1);
  int carry = 0;
  size_t n = std::max(a.size(), b.size());
  for (size_t i = 0; i < n || carry; ++i) {
    long long cur = carry;
    if (i < a.size()) cur += a[i];
    if (i < b.size()) cur += b[i];
    res.push_back((int)(cur % base));
    carry = (int)(cur / base);
  }
  return res;
}

static std::vector<int> sub_abs(const std::vector<int> &a, const std::vector<int> &b) {
  // assume |a| >= |b|
  const int base = BigIntCore::base;
  std::vector<int> res;
  res.reserve(a.size());
  int carry = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    long long cur = a[i] - carry - (i < b.size() ? b[i] : 0);
    if (cur < 0) { cur += base; carry = 1; } else carry = 0;
    res.push_back((int)cur);
  }
  trim(res);
  return res;
}

// FFT-based multiplication (Cooley-Tukey) using long double complex
struct C { long double x, y; C(long double X=0, long double Y=0):x(X),y(Y){} };
static C operator+(const C &a, const C &b){ return C(a.x+b.x, a.y+b.y);}
static C operator-(const C &a, const C &b){ return C(a.x-b.x, a.y-b.y);}
static C operator*(const C &a, const C &b){ return C(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);}

static void fft(std::vector<C> &a, bool invert) {
  int n = (int)a.size();
  // bit-reverse permutation
  for (int i=1, j=0; i<n; ++i) {
    int bit = n>>1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i<j) std::swap(a[i], a[j]);
  }
  for (int len=2; len<=n; len<<=1) {
    long double ang = 2.0L * 3.14159265358979323846264338327950288L / len * (invert ? -1.0L : 1.0L);
    C wlen(std::cos(ang), std::sin(ang));
    for (int i=0; i<n; i+=len) {
      C w(1,0);
      for (int j=0; j<len/2; ++j) {
        C u = a[i+j];
        C v = a[i+j+len/2] * w;
        a[i+j] = u + v;
        a[i+j+len/2] = u - v;
        w = w * wlen;
      }
    }
  }
  if (invert) {
    for (int i=0; i<n; ++i) { a[i].x /= n; a[i].y /= n; }
  }
}

static std::vector<int> mul_abs_fft(const std::vector<int> &a, const std::vector<int> &b) {
  // multiply arrays of base=1e4 digits
  std::vector<C> fa, fb;
  int n = 1;
  while (n < (int)a.size() + (int)b.size()) n <<= 1;
  fa.resize(n); fb.resize(n);
  for (size_t i=0;i<a.size();++i) fa[i].x = a[i];
  for (size_t i=a.size(); i<(size_t)n; ++i) fa[i] = C();
  for (size_t i=0;i<b.size();++i) fb[i].x = b[i];
  for (size_t i=b.size(); i<(size_t)n; ++i) fb[i] = C();
  fft(fa,false); fft(fb,false);
  for (int i=0;i<n;++i) fa[i] = fa[i]*fb[i];
  fft(fa,true);
  std::vector<int> res(n);
  long long carry = 0;
  const int base = BigIntCore::base;
  for (int i=0;i<n;++i) {
    long long t = (long long)std::llround(fa[i].x) + carry;
    res[i] = (int)(t % base);
    carry = t / base;
  }
  while (carry) { res.push_back((int)(carry % base)); carry /= base; }
  trim(res);
  return res;
}

class BigIntImpl {
public:
  bool neg = false;
  std::vector<int> d; // little-endian base 1e4

  BigIntImpl() {}
};

// Internal helpers bound to int2048
static BigIntImpl &impl_of(int2048 &x) { return *(BigIntImpl*)&x; }
static const BigIntImpl &impl_of(const int2048 &x) { return *(const BigIntImpl*)&x; }

// Constructors
int2048::int2048() {
  BigIntImpl &I = impl_of(*this); I.neg = false; I.d.clear();
}

int2048::int2048(long long v) {
  BigIntImpl &I = impl_of(*this);
  I.neg = (v < 0);
  unsigned long long u = I.neg ? (unsigned long long)(-(v + 1)) + 1ULL : (unsigned long long)v;
  const int base = BigIntCore::base;
  while (u) { I.d.push_back((int)(u % base)); u /= base; }
}

int2048::int2048(const std::string &s) { BigIntImpl &I = impl_of(*this); I.neg=false; I.d.clear(); read(s); }

int2048::int2048(const int2048 &o) {
  const BigIntImpl &O = impl_of(o);
  BigIntImpl &I = impl_of(*this);
  I.neg = O.neg; I.d = O.d;
}

void int2048::read(const std::string &s) {
  BigIntImpl &I = impl_of(*this);
  I.neg=false; I.d.clear();
  int pos = 0; while (pos<(int)s.size() && (s[pos]==' '||s[pos]=='\n' || s[pos]=='\t')) ++pos;
  if (pos<(int)s.size() && (s[pos]=='+'||s[pos]=='-')) { I.neg = (s[pos]=='-'); ++pos; }
  while (pos<(int)s.size() && s[pos]=='0') ++pos;
  std::vector<int> tmp;
  const int base = BigIntCore::base;
  const int bd = BigIntCore::base_digits;
  for (int i = (int)s.size(); i > pos; i -= bd) {
    int x = 0;
    int l = std::max(pos, i - bd);
    for (int j=l; j<i; ++j) x = x*10 + (s[j]-'0');
    tmp.push_back(x);
  }
  // tmp is big-endian chunks reversed; convert to little-endian
  I.d = tmp;
  trim(I.d);
  if (I.d.empty()) I.neg=false; // zero has no sign
}

void int2048::print() {
  const BigIntImpl &I = impl_of(*this);
  if (I.d.empty()) { std::cout << 0; return; }
  if (I.neg) std::cout << '-';
  int bd = BigIntCore::base_digits;
  std::cout << I.d.back();
  for (int i=(int)I.d.size()-2; i>=0; --i) {
    char buf[16]; std::snprintf(buf, sizeof(buf), "%0*d", bd, I.d[i]);
    std::cout << buf;
  }
}

int2048 &int2048::add(const int2048 &b) {
  BigIntImpl &A = impl_of(*this);
  const BigIntImpl &B = impl_of(b);
  if (A.neg == B.neg) {
    A.d = add_abs(A.d, B.d);
  } else {
    int c = cmp_abs(A.d, B.d);
    if (c >= 0) {
      A.d = sub_abs(A.d, B.d);
    } else {
      A.d = sub_abs(B.d, A.d);
      A.neg = B.neg;
    }
  }
  if (A.d.empty()) A.neg=false;
  return *this;
}

int2048 add(int2048 a, const int2048 &b) { a.add(b); return a; }

int2048 &int2048::minus(const int2048 &b) {
  BigIntImpl &A = impl_of(*this);
  const BigIntImpl &B = impl_of(b);
  if (A.neg != B.neg) {
    A.d = add_abs(A.d, B.d);
  } else {
    int c = cmp_abs(A.d, B.d);
    if (c >= 0) {
      A.d = sub_abs(A.d, B.d);
    } else {
      A.d = sub_abs(B.d, A.d);
      A.neg = !A.neg;
    }
  }
  if (A.d.empty()) A.neg=false;
  return *this;
}

int2048 minus(int2048 a, const int2048 &b) { a.minus(b); return a; }

int2048 int2048::operator+() const { return *this; }

int2048 int2048::operator-() const {
  int2048 r(*this);
  BigIntImpl &R = impl_of(r);
  if (!R.d.empty()) R.neg = !R.neg;
  return r;
}

int2048 &int2048::operator=(const int2048 &o) {
  const BigIntImpl &O = impl_of(o);
  BigIntImpl &I = impl_of(*this);
  if (this == &o) return *this;
  I.neg = O.neg; I.d = O.d; return *this;
}

int2048 &int2048::operator+=(const int2048 &b) { return add(b); }
int2048 operator+(int2048 a, const int2048 &b) { return a += b; }

int2048 &int2048::operator-=(const int2048 &b) { return minus(b); }
int2048 operator-(int2048 a, const int2048 &b) { return a -= b; }

int2048 &int2048::operator*=(const int2048 &b) {
  BigIntImpl &A = impl_of(*this);
  const BigIntImpl &B = impl_of(b);
  if (A.d.empty() || B.d.empty()) { A.d.clear(); A.neg=false; return *this; }
  // choose method
  std::vector<int> res;
  size_t n = A.d.size(), m = B.d.size();
  if (std::min(n,m) < 64) {
    const int base = BigIntCore::base;
    res.assign(n + m, 0);
    for (size_t i=0;i<n;++i) {
      long long carry = 0;
      for (size_t j=0;j<m || carry;++j) {
        long long cur = res[i+j] + (long long)A.d[i] * (j<m ? B.d[j] : 0) + carry;
        res[i+j] = (int)(cur % base);
        carry = cur / base;
      }
    }
    trim(res);
  } else {
    res = mul_abs_fft(A.d, B.d);
  }
  A.d.swap(res);
  A.neg = (A.neg != B.neg);
  if (A.d.empty()) A.neg=false;
  return *this;
}

int2048 operator*(int2048 a, const int2048 &b) { return a *= b; }

// Division and modulo: floor division with Python semantics
static void div_mod_abs(const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &q, std::vector<int> &r) {
  const int base = BigIntCore::base;
  r.clear(); q.clear();
  if (b.empty()) return; // undefined; tests guarantee not 0
  if (cmp_abs(a,b) < 0) { r = a; return; }
  int n = (int)a.size(), m = (int)b.size();
  q.assign(n - m + 1, 0);
  // Normalized long division
  int norm = base / (b.back() + 1);
  std::vector<int> A(a), B(b);
  // multiply by norm
  long long carry = 0;
  for (int i=0;i<n;++i) { long long cur = (long long)A[i]*norm + carry; A[i] = (int)(cur % base); carry = cur / base; }
  if (carry) A.push_back((int)carry), ++n;
  carry = 0;
  for (int i=0;i<m;++i) { long long cur = (long long)B[i]*norm + carry; B[i] = (int)(cur % base); carry = cur / base; }
  if (carry) B.push_back((int)carry), ++m;
  std::vector<int> R(n,0);
  for (int i=0;i<n;++i) R[i] = A[i];
  std::vector<int> Bb = B;
  // shift-based division
  for (int i = n - 1; i >= m - 1; --i) {
    long long r2 = (long long)R[i];
    long long r1 = (i-1>=0)? (long long)R[i-1] : 0;
    long long r0 = (i-2>=0)? (long long)R[i-2] : 0; // help estimation stability
    long long est = (r2*base + r1) / B.back();
    if (est >= base) est = base-1;
    // subtract est * B shifted by (i - m + 1)
    int k = i - (m - 1);
    long long borrow = 0;
    for (int j=0; j<m || borrow; ++j) {
      long long cur = (long long)R[j+k] - (long long)(j<m ? B[j] : 0) * est - borrow;
      if (cur < 0) { long long db = (-cur + base - 1)/base; cur += db * base; borrow = db; } else borrow = 0;
      R[j+k] = (int)cur;
    }
    // fix if negative by adding back B and decrementing est
    // check if leading segment became negative by comparing top digits
    while (true) {
      // If R segment >= 0 we are fine; ensure no over-borrow propagated beyond k+m-1
      bool negSeg = false;
      for (int t=i; t>=i-2 && t>=0; --t) if (R[t]<0) { negSeg=true; break; }
      if (!negSeg) break;
      // add back B shifted
      long long carry2 = 0;
      for (int j=0; j<m || carry2; ++j) {
        long long cur = (long long)R[j+k] + (long long)(j<m ? B[j] : 0) + carry2;
        R[j+k] = (int)(cur % base);
        carry2 = cur / base;
      }
      --est;
    }
    q[k] = (int)est;
  }
  // unnormalize remainder
  // R has length n; the remainder is first m-1 digits
  r.assign(R.begin(), R.begin() + (m-1));
  // divide by norm
  long long carry3 = 0;
  for (int i=(int)r.size()-1; i>=0; --i) {
    long long cur = r[i] + carry3 * base;
    r[i] = (int)(cur / norm);
    carry3 = cur % norm;
  }
  trim(q); trim(r);
}

int2048 &int2048::operator/=(const int2048 &b) {
  BigIntImpl &A = impl_of(*this);
  const BigIntImpl &B = impl_of(b);
  if (B.d.empty()) return *this; // undefined
  if (A.d.empty()) { A.neg=false; return *this; }
  std::vector<int> q, r;
  // compute abs division
  div_mod_abs(A.d, B.d, q, r);
  bool negQ = (A.neg != B.neg);
  // Python floor division adjustment: if negQ and r != 0, decrement q by 1 and adjust r
  if (!r.empty()) {
    if (A.neg != B.neg) {
      // q = q - 1; r = r + |B|
      // decrement q
      const int base = BigIntCore::base;
      int i=0; while (i<(int)q.size()) {
        if (q[i]>0) { --q[i]; break; }
        q[i] = base-1; ++i;
      }
      trim(q);
      // r = r + |B|
      r = add_abs(r, B.d);
    }
  }
  A.d.swap(q);
  A.neg = (!A.d.empty() ? negQ : false);
  return *this;
}

int2048 operator/(int2048 a, const int2048 &b) { return a /= b; }

int2048 &int2048::operator%=(const int2048 &b) {
  BigIntImpl &A = impl_of(*this);
  const BigIntImpl &B = impl_of(b);
  if (B.d.empty()) return *this; // undefined
  if (A.d.empty()) { A.neg=false; return *this; }
  std::vector<int> q, r;
  div_mod_abs(A.d, B.d, q, r);
  bool negQ = (A.neg != B.neg);
  if (!r.empty()) {
    if (A.neg != B.neg) {
      r = add_abs(r, B.d);
    }
  }
  A.d.swap(r);
  // modulo sign follows divisor
  A.neg = (!A.d.empty() ? B.neg : false);
  return *this;
}

int2048 operator%(int2048 a, const int2048 &b) { return a %= b; }

std::istream &operator>>(std::istream &in, int2048 &x) {
  std::string s; in >> s; x.read(s); return in;
}

std::ostream &operator<<(std::ostream &out, const int2048 &x) {
  const BigIntImpl &I = impl_of(x);
  if (I.d.empty()) { out << 0; return out; }
  if (I.neg) out << '-';
  int bd = BigIntCore::base_digits;
  out << I.d.back();
  for (int i=(int)I.d.size()-2; i>=0; --i) {
    char buf[16]; std::snprintf(buf, sizeof(buf), "%0*d", bd, I.d[i]);
    out << buf;
  }
  return out;
}

bool operator==(const int2048 &a, const int2048 &b) {
  const BigIntImpl &A = impl_of(a); const BigIntImpl &B = impl_of(b);
  return A.neg==B.neg && A.d==B.d;
}

bool operator!=(const int2048 &a, const int2048 &b) { return !(a==b); }

bool operator<(const int2048 &a, const int2048 &b) {
  const BigIntImpl &A = impl_of(a); const BigIntImpl &B = impl_of(b);
  if (A.neg != B.neg) return A.neg;
  int c = cmp_abs(A.d, B.d);
  return A.neg ? (c>0) : (c<0);
}

bool operator>(const int2048 &a, const int2048 &b) { return b < a; }
bool operator<=(const int2048 &a, const int2048 &b) { return !(b<a); }
bool operator>=(const int2048 &a, const int2048 &b) { return !(a<b); }

} // namespace sjtu
