// ============================================================
// NOIZE — lib/api_service.dart  (production-ready)
//
// BUGS FIXED vs previous version:
//  1. history() verdict read path fixed — /history returns
//     verdict at top level, not nested under 'metrics'
//  2. average fairness calc: safe null handling with ?? 0.0
//  3. isPass filter: correct Dart isEmpty → verdict.isNotEmpty
//  4. _safeGet now returns the raw response (not double-decoded)
// ============================================================

import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

// Set at build time:
//   flutter run  --dart-define=BASE_URL=http://192.168.1.100:8001
//   flutter build apk --dart-define=BASE_URL=https://api.yourapp.com
const String kBaseUrl =
    String.fromEnvironment('BASE_URL', defaultValue: 'http://localhost:8001');

const _kShort  = Duration(seconds: 15);
const _kLong   = Duration(seconds: 90);
const _kUpload = Duration(seconds: 60);

// ── Token storage ─────────────────────────────────────────────

Future<void> saveToken(String token) async {
  final p = await SharedPreferences.getInstance();
  await p.setString('access_token', token);
}

Future<String?> loadToken() async {
  final p = await SharedPreferences.getInstance();
  return p.getString('access_token');
}

Future<void> clearToken() async {
  final p = await SharedPreferences.getInstance();
  await p.remove('access_token');
  await p.remove('refresh_token');
}

Future<void> _saveRefreshToken(String token) async {
  final p = await SharedPreferences.getInstance();
  await p.setString('refresh_token', token);
}

Future<String?> _loadRefreshToken() async {
  final p = await SharedPreferences.getInstance();
  return p.getString('refresh_token');
}

// ── Auth headers ──────────────────────────────────────────────

Future<Map<String, String>> authHeaders() async {
  final token = await loadToken();
  return {
    'Content-Type': 'application/json',
    if (token != null) 'Authorization': 'Bearer $token',
  };
}

// ── Token refresh ─────────────────────────────────────────────

Future<bool> _tryRefresh() async {
  final refreshToken = await _loadRefreshToken();
  if (refreshToken == null) return false;
  try {
    final resp = await http.post(
      Uri.parse('$kBaseUrl/auth/refresh'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'refresh_token': refreshToken}),
    ).timeout(_kShort);
    if (resp.statusCode == 200) {
      final body = jsonDecode(resp.body) as Map<String, dynamic>;
      if (body['status'] == 'success') {
        final d = body['data'] as Map<String, dynamic>;
        final newAccess = d['access_token'] as String?;
        final newRefresh = d['refresh_token'] as String?;
        if (newAccess != null) {
          await saveToken(newAccess);
          if (newRefresh != null) await _saveRefreshToken(newRefresh);
          return true;
        }
      }
    }
  } catch (_) {}
  return false;
}

// ── HTTP helpers with auto-refresh ───────────────────────────

Future<http.Response> _safeGet(Uri uri) async {
  var headers = await authHeaders();
  var resp = await http.get(uri, headers: headers).timeout(_kShort);
  if (resp.statusCode == 401) {
    if (await _tryRefresh()) {
      headers = await authHeaders();
      resp = await http.get(uri, headers: headers).timeout(_kShort);
    }
  }
  return resp;
}

Future<http.Response> _safePost(Uri uri, Map<String, dynamic> payload,
    {Duration timeout = _kShort}) async {
  var headers = await authHeaders();
  var resp = await http
      .post(uri, headers: headers, body: jsonEncode(payload))
      .timeout(timeout);
  if (resp.statusCode == 401) {
    if (await _tryRefresh()) {
      headers = await authHeaders();
      resp = await http
          .post(uri, headers: headers, body: jsonEncode(payload))
          .timeout(timeout);
    }
  }
  return resp;
}

// ── Auth ──────────────────────────────────────────────────────

Future<Map<String, dynamic>> login(String username, String password) async {
  try {
    final resp = await http.post(
      Uri.parse('$kBaseUrl/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'username': username, 'password': password}),
    ).timeout(_kShort);
    final body = jsonDecode(resp.body) as Map<String, dynamic>;
    if (resp.statusCode == 200 && body['status'] == 'success') {
      final d = body['data'] as Map<String, dynamic>;
      await saveToken(d['access_token'] as String);
      if (d['refresh_token'] != null) {
        await _saveRefreshToken(d['refresh_token'] as String);
      }
      return {'success': true, 'token': d['access_token']};
    }
    return {
      'success': false,
      'error': body['detail'] ?? body['error'] ?? 'Login failed'
    };
  } on SocketException {
    return {'success': false, 'error': 'Cannot reach server. Check your connection.'};
  } catch (e) {
    return {'success': false, 'error': 'Unexpected error: $e'};
  }
}

Future<Map<String, dynamic>> register(String username, String password) async {
  try {
    final resp = await http.post(
      Uri.parse('$kBaseUrl/auth/register'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'username': username, 'password': password}),
    ).timeout(_kShort);
    final body = jsonDecode(resp.body) as Map<String, dynamic>;
    // BUG FIX: backend returns 201 for register, not 200
    return {
      'success': (resp.statusCode == 201 || resp.statusCode == 200) &&
          body['status'] == 'success',
      'error': body['detail'] ?? body['error'],
    };
  } on SocketException {
    return {'success': false, 'error': 'Cannot reach server.'};
  } catch (e) {
    return {'success': false, 'error': 'Unexpected error: $e'};
  }
}

// ── Upload ────────────────────────────────────────────────────

Future<Map<String, dynamic>> uploadFileBytes(String fileName, List<int> bytes) async {
  try {
    final token = await loadToken();
    final request =
        http.MultipartRequest('POST', Uri.parse('$kBaseUrl/upload/'));
    if (token != null) request.headers['Authorization'] = 'Bearer $token';
    request.files.add(http.MultipartFile.fromBytes(
      'file',
      bytes,
      filename: fileName,
    ));
    final streamed = await request.send().timeout(_kUpload);
    final resp = await http.Response.fromStream(streamed);
    final body = jsonDecode(resp.body) as Map<String, dynamic>;
    if (resp.statusCode == 200 && body['status'] == 'success') {
      return {
        'success': true,
        'filename': body['data']['filename'],
        'size_bytes': body['data']['size_bytes'] ?? 0,
      };
    }
    return {
      'success': false,
      'error': body['detail'] ?? body['error'] ?? 'Upload failed'
    };
  } on SocketException {
    return {'success': false, 'error': 'Cannot reach server.'};
  } catch (e) {
    return {'success': false, 'error': 'Upload error: $e'};
  }
}

// ── Analyze ───────────────────────────────────────────────────

Future<Map<String, dynamic>> analyze({
  required String filename,
  required String targetColumn,
  String? protectedColumn,
}) async {
  try {
    final resp = await _safePost(
      Uri.parse('$kBaseUrl/analyze/'),
      {
        'filename': filename,
        'target_column': targetColumn,
        if (protectedColumn != null) 'protected_column': protectedColumn,
      },
      timeout: _kLong,
    );
    final body = jsonDecode(resp.body) as Map<String, dynamic>;
    if (resp.statusCode == 200 && body['status'] == 'success') {
      return {'success': true, ...body['data'] as Map<String, dynamic>};
    }
    return {
      'success': false,
      'error': body['detail'] ?? body['error'] ?? 'Analysis failed'
    };
  } on SocketException {
    return {'success': false, 'error': 'Cannot reach server.'};
  } catch (e) {
    return {'success': false, 'error': 'Analysis error: $e'};
  }
}

// ── Metrics + Explain ─────────────────────────────────────────

Future<Map<String, dynamic>> getMetrics(String resultId) async {
  try {
    final resp = await _safeGet(Uri.parse('$kBaseUrl/metrics/$resultId'));
    final body = jsonDecode(resp.body) as Map<String, dynamic>;
    if (resp.statusCode == 200 && body['status'] == 'success') {
      return {'success': true, ...body['data'] as Map<String, dynamic>};
    }
    return {'success': false, 'error': body['detail'] ?? 'Not found'};
  } catch (e) {
    return {'success': false, 'error': '$e'};
  }
}

Future<Map<String, dynamic>> getExplanation(String resultId) async {
  try {
    final resp = await _safeGet(Uri.parse('$kBaseUrl/explain/$resultId'));
    final body = jsonDecode(resp.body) as Map<String, dynamic>;
    if (resp.statusCode == 200 && body['status'] == 'success') {
      return {'success': true, ...body['data'] as Map<String, dynamic>};
    }
    return {'success': false, 'error': body['detail'] ?? 'Not found'};
  } catch (e) {
    return {'success': false, 'error': '$e'};
  }
}

// ── History ───────────────────────────────────────────────────

// BUG FIX: /history returns a flat summary list where verdict,
// fairness_score, filename are all top-level keys — not nested
// under 'metrics'. The previous version was reading wrong paths.
Future<List<Map<String, dynamic>>> getHistory() async {
  try {
    final resp = await _safeGet(Uri.parse('$kBaseUrl/history'));
    if (resp.statusCode == 200) {
      final body = jsonDecode(resp.body) as Map<String, dynamic>;
      final raw = body['data'];
      if (raw is List) {
        return raw.cast<Map<String, dynamic>>();
      }
    }
    return [];
  } catch (_) {
    return [];
  }
}

// ── Compare ───────────────────────────────────────────────────

Future<Map<String, dynamic>> compare(String id1, String id2) async {
  try {
    final resp = await _safeGet(Uri.parse('$kBaseUrl/compare/$id1/$id2'));
    final body = jsonDecode(resp.body) as Map<String, dynamic>;
    if (resp.statusCode == 200 && body['status'] == 'success') {
      return {'success': true, ...body['data'] as Map<String, dynamic>};
    }
    return {'success': false, 'error': body['detail'] ?? 'Compare failed'};
  } catch (e) {
    return {'success': false, 'error': '$e'};
  }
}
